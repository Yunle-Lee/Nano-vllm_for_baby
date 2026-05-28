import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS Normalization：Qwen3 使用的归一化层。
    
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    
    优势（vs LayerNorm）：
    - 不做均值中心化（无 bias 减）
    - 计算更简单，速度更快
    
    提供两种变体：
    1. rms_forward: 纯 norm
    2. add_rms_forward: fused add + norm（残差连接 + 归一化融合）
       将 residual addition 和 RMSNorm 合并在一个 kernel 中，减少显存读写
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放因子

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """纯 RMSNorm。
        
        过程：x → float32 → rms → scale → 转回原 dtype
        中间使用 float32 是为了数值精度。
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)    # 均方值
        x.mul_(torch.rsqrt(var + self.eps))          # x / sqrt(var + eps)
        x = x.to(orig_dtype).mul_(self.weight)        # scale
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused Add + RMSNorm：将残差连接和 norm 融合为一个 kernel。
        
        Qwen3 的 parallel residual 设计：
        input_layernorm(hidden, residual) → attention...
        post_attention_layernorm(hidden, residual) → mlp...
        
        每个 norm 同时做 残差加法 + 归一化：
        1. x = x + residual
        2. 将结果赋值给 residual（传给下一个 norm 用）
        3. 对 x 做 RMSNorm
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())         # 残差连接
        residual = x.to(orig_dtype)                   # 保存残差（给下一个 norm 用）
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        """如果没有 residual，做纯 norm；有 residual，做 fused add+norm"""
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
