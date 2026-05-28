import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 的注意力模块。
    
    数据流：
    hidden_states → QKV proj (并行切分) → split Q, K, V
      → QK LayerNorm (如果无 bias)
      → RoPE (对 Q, K 旋转位置编码)
      → PagedAttention (带 KV cache 的 attention)
      → O proj (行并行，输出层)
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, max_position=4096*32,
                 head_dim=None, rms_norm_eps=1e-06, qkv_bias=False, rope_theta=10000,
                 rope_scaling=None):
        super().__init__()
        tp_size = dist.get_world_size()

        # Head 数在 TP 下按 GPU 数均匀切分
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size              # 本 rank 的 Q head 数

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size        # 本 rank 的 KV head 数

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim                  # Q 的总维度（本 rank）
        self.kv_size = self.num_kv_heads * self.head_dim              # KV 的总维度（本 rank）
        self.scaling = self.head_dim ** -0.5                          # attention scale = 1/sqrt(d_k)
        self.qkv_bias = qkv_bias

        # Q、K、V 的线性投影合并为一个列并行层
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # O 投影（行并行：每个 GPU 算自己那部分，然后 all_reduce）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False,
        )

        # RoPE 的位置编码参数（从 rope_scaling 字典中提取）
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

        # Qwen3 无 bias 的注意力使用 QK-Norm（在 RoPE 之前对 Q、K 做 LayerNorm）
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)   # 对每个 head 做归一化
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1) QKV 投影（列并行，各 GPU 计算自己的分片）
        qkv = self.qkv_proj(hidden_states)                           # [N, q_size + 2*kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2) 重塑为多 head 格式
        q = q.view(-1, self.num_heads, self.head_dim)               # [N, num_heads, head_dim]
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 3) QK-Norm（仅在无 bias 模式下，Qwen3 特有设计）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 4) 旋转位置编码（RoPE）
        q, k = self.rotary_emb(positions, q, k)

        # 5) PagedAttention（含 KV cache 存储 + flash_attn 计算）
        o = self.attn(q, k, v)

        # 6) O 投影（行并行）+ all_reduce
        output = self.o_proj(o.flatten(1, -1))                       # [N, hidden_size]
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 的 MLP 模块（SwiGLU 架构）。
    
    数据流：
    hidden_states → gate_up_proj（列并行，输出 gate 和 up 两个部分）
      → SiluAndMul（SiLU(gate) * up）
      → down_proj（行并行，all_reduce）
    """

    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        # gate + up 合并为一个列并行层（输出 = 2 * intermediate_size/tp）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
        )
        # down 投影（行并行）
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"              # Qwen3 只用 SiLU 激活
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)           # [N, 2*intermediate_size/tp]
        x = self.act_fn(gate_up)                 # SiLU(gate) * up
        x = self.down_proj(x)                    # [N, hidden_size]（跨 GPU all_reduce）
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 的解码器层。
    
    采用 parallel residual 设计（残差连接从一层传到下一层，而非在层内循环）：
    
    residual 流向：
    Layer N:   residual_N → input_layernorm → attention → post_attention_layernorm → mlp → (hidden, residual_{N+1})
    Layer N+1: residual_{N+1} → ...
    
    这与传统 Transformer 不同，传统的 residual 是在每一层内部保存和使用的。
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        """Decoder Layer 前向传播。
        
        关键：第一个 decoder layer 的 residual 为 None（不做 add+norm 中的 add），
        后续 layer 的 residual 由上一层传入，做 fused add+rmsnorm。
        """
        # 1) Pre-attention RMSNorm（第一层不做残差加法）
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # 2) Self-Attention
        hidden_states = self.self_attn(positions, hidden_states)

        # 3) Post-attention RMSNorm（fused add+norm）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # 4) MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 主体模型（不含 LM Head）。
    
    数据流：
    input_ids → VocabParallelEmbedding → N × DecoderLayer → final RMSNorm → hidden_states
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层的输出做 final RMSNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 因果语言模型（完整模型，含 LM Head）。
    
    packed_modules_mapping 的作用：
    HuggingFace 保存的权重文件中，q_proj/k_proj/v_proj 是分开的三个 tensor，
    但在 TP 模式下它们被合并为一个 QKVParallelLinear。
    同样 gate_proj/up_proj 被合并为 MergedColumnParallelLinear。
    
    这个映射表告诉 weight loader 如何将 HF 的分散权重填入合并后的参数：
    - "q_proj" → ("qkv_proj", "q")：HF 的 q_proj.weight 映射到模型的 qkv_proj.weight 的 Q 部分
    - "gate_proj" → ("gate_up_proj", 0)：HF 的 gate_proj.weight 映射到 gate_up_proj.weight 的第 0 份
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),       # HF 固有名称 → (模型参数名, 分片标识)
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),   # 数字表示在 merged 参数中的偏移索引
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # Weight Tying：如果配置要求，将 LM Head 的权重与 embedding 共享
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """完整前向：embedding → transformer → 返回 hidden_states（不含 logits 计算）"""
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """从 hidden_states 计算 logits（LM Head）。
        
        分离 forward 和 compute_logits 的原因：
        - 在 CUDA Graph 捕获时，只录制 forward 部分（transformer body），
          compute_logits 在 graph 外执行（因为它包含 TP gather 操作，不友好于 graph）
        - 在 prefill 阶段，只取最后一个 token 的 hidden_states 计算 logits
        """
        return self.lm_head(hidden_states)
