from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """应用旋转位置编码（RoPE）。
    
    数学：对 x 的前后两半进行 2D 旋转变换
    x1, x2 = chunk(x, 2, dim=-1)
    y1 = x1*cos - x2*sin
    y2 = x2*cos + x1*sin
    
    为什么要分两半？
    RoPE 将 head_dim 维度的每对相邻元素看作一个 2D 向量，
    按位置信息旋转。前半和后半各代表一对。
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)     # 前后各 head_dim/2
    y1 = x1 * cos - x2 * sin                        # 旋转变换第一半
    y2 = x2 * cos + x1 * sin                        # 旋转变换第二半
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码模块。
    
    预计算所有位置的 cos/sin 并缓存，避免每次前向都重新计算。
    cos_sin_cache shape: [max_position, 1, head_dim]
    第 1 维(=1) 用于广播到 batch 维度。
    """

    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size               # nano-vllm 目前只支持 full RoPE

        # 预计算频率：inv_freq[i] = 1 / (base^(2i/rotary_dim))
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 每个位置的旋转角度：freqs[pos, i] = pos * inv_freq[i]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)    # [max_pos, head_dim/2]

        cos = freqs.cos()
        sin = freqs.sin()
        # 拼接 cos 和 sin 为一个 tensor：[max_pos, 1, head_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # persistent=False 表示这是缓冲区（非参数），不参与 state_dict 保存
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        """应用 RoPE 到 query 和 key。
        
        Args:
            positions: [total_tokens] 每个 token 的绝对位置
            query: [total_tokens, num_heads, head_dim]
            key:   [total_tokens, num_kv_heads, head_dim]
        Returns:
            rotated query, rotated key
        """
        cos_sin = self.cos_sin_cache[positions]       # 查表获取对应位置的 cos, sin
        cos, sin = cos_sin.chunk(2, dim=-1)           # 拆分 cos 和 sin
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float):
    """工厂函数：获取 RoPE 模块（带 LRU 缓存，相同参数复用）"""
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
