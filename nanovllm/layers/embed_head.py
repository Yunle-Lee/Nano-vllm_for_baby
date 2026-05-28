import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表并行嵌入层。
    
    将词表按 TP size 均匀切分到多个 GPU：
    - GPU 0: token 0 ~ vocab//tp_size - 1
    - GPU 1: token vocab//tp_size ~ 2*vocab//tp_size - 1
    - ...
    
    Forward 流程：
    1. 将每个 token ID 映射到所在 GPU 的局部 vocab 索引
    2. 只对属于本 GPU 的 token 做 embedding lookup
    3. 其他 token 置零
    4. all_reduce 汇总得到完整 embedding
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size   # 本 GPU 持有的词表大小
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank   # 本 GPU 负责的第一个 token ID
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader  # 绑定 TP 权重加载器

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """TP 切分加载词表权重"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """前向传播。
        
        Args:
            x: token IDs [total_tokens], dtype int64
        Returns:
            embeddings: [total_tokens, embedding_dim]
        """
        if self.tp_size > 1:
            # 创建 mask：只对属于本 rank 的 token 做 embedding
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将 token ID 映射到本 GPU 的局部索引（不在此范围内的置 0）
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y             # 将不需要的 token 的 embedding 置零
            dist.all_reduce(y)                     # 各 GPU 汇总（等价于对 mask 做 all_reduce）
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """并行 LM Head：将 hidden_states 映射到完整词表上的 logits。
    
    与 VocabParallelEmbedding 共享权重布局（切分方式一致）。
    区别在于：
    - Embedding 是 lookup（token → hidden）
    - LM Head 是线性变换（hidden → logits）
    
    Prefill 阶段只计算每条序列最后一个 token 的 logits（中间的 tokens 不需要采样）。
    Decode 阶段每条序列只有 1 个新 token，所以都计算。
    """

    def __init__(self, num_embeddings, embedding_dim, bias=False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """计算 logits。
        
        Args:
            x: [total_tokens, hidden_size]
        Returns:
            logits: [num_seqs, vocab_size]（只有 rank 0 有完整 logits）
        """
        context = get_context()
        if context.is_prefill:
            # Prefill：多条序列的多个 token 混合在一起
            # 只取每条序列的最后一个 token（cu_seqlens_q[i+1]-1 是第 i 条序列最后一个 token 的索引）
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()       # [num_seqs, hidden_size]

        # 每个 GPU 计算局部的 logits
        logits = F.linear(x, self.weight)          # [num_seqs, vocab_size/tp]

        if self.tp_size > 1:
            # gather：收集所有 GPU 的局部 logits，只 rank 0 持有完整结果
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)      # 从所有 rank gather 到 rank 0
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None  # 拼接完整 logits
        return logits
