import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,          # 指向新计算的 key tensor
    key_stride,       # key tensor 的行步长
    value_ptr,        # 指向新计算的 value tensor
    value_stride,     # value tensor 的行步长
    k_cache_ptr,      # 指向 GPU 上的 K cache
    v_cache_ptr,      # 指向 GPU 上的 V cache
    slot_mapping_ptr, # 每个 token 应写入 KV cache 的位置索引
    D: tl.constexpr,  # num_heads * head_dim（展开为一维）
):
    """Triton kernel：将新计算的 K、V 写入 GPU KV cache 的对应 slot。
    
    每个 program instance 处理一个 token 的 key/value 写入。
    slot_mapping[i] 告诉这个 token 的 KV 应该写到 KV cache 的哪个位置。
    slot_mapping[i] == -1 表示 padding，跳过。
    """
    idx = tl.program_id(0)                         # 当前处理的 token 索引
    slot = tl.load(slot_mapping_ptr + idx)         # 读取该 token 的写入位置
    if slot == -1:
        return                                      # padding token，跳过

    # 计算 key 在 buffer 中的起始地址
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 从临时 buffer 中加载 key/value
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 写入 KV cache 对应 slot
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
                  v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """将本轮计算的 Key/Value 存入 KV cache。
    
    内存布局说明：
    - key/value shape: [N, num_heads, head_dim]
    - k_cache/v_cache shape: [num_blocks * block_size, num_heads * head_dim]
    即 KV cache 将 head_dim 维度展开存储。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 确保内存布局连续，便于 kernel 中直接按偏移访问
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 N 个 triton kernel instance
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """PagedAttention 注意力层。
    
    职责：
    1. 将新计算的 K、V 存入 KV cache（通过 Triton kernel）
    2. 调用 flash_attn 的不同接口完成 attention 计算
    
    Prefill 和 Decode 使用不同的 flash_attn 接口：
    - Prefill: flash_attn_varlen_func（变长序列，支持 block_table 前缀缓存索引）
    - Decode:  flash_attn_with_kvcache（Q 长度=1 的优化路径）
    
    k_cache / v_cache 引用 GPU 上的大 tensor，由 ModelRunner.allocate_kv_cache() 注入。
    """

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads               # 当前 TP rank 的 Q head 数
        self.head_dim = head_dim
        self.scale = scale                        # 1/sqrt(head_dim)
        self.num_kv_heads = num_kv_heads         # 当前 TP rank 的 KV head 数
        self.k_cache = self.v_cache = torch.tensor([])  # 占位，由 allocate_kv_cache 注入

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Attention 前向计算。
        
        Args:
            q: Query [total_q_tokens, num_heads, head_dim]
            k: Key   [total_kv_tokens, num_kv_heads, head_dim]
            v: Value [total_kv_tokens, num_kv_heads, head_dim]
        
        步骤：
        1. 将本轮 K、V 写入 KV cache
        2. 根据 prefill/decode 阶段选择不同的 flash_attn 接口
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # ========== 第 1 步：存储 K、V 到 KV cache ==========
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # ========== 第 2 步：Attention 计算 ==========
        if context.is_prefill:
            # Prefill 阶段
            if context.block_tables is not None:  # 有前缀缓存命中
                k, v = k_cache, v_cache           # 用完整 KV cache 做 attention
            # varlen 支持变长序列（不同序列有不同长度）
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,                      # 因果遮罩
                block_table=context.block_tables  # 前缀缓存需要的跨块索引
            )
        else:
            # Decode 阶段：Q 长度=1，直接从 KV cache 读取
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),                   # [N] → [N, 1, num_heads, head_dim]
                k_cache, v_cache,
                cache_seqlens=context.context_lens,  # 每条序列的 KV 长度
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        return o
