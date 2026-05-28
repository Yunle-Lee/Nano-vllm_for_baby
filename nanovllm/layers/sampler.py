import torch
from torch import nn


class Sampler(nn.Module):
    """采样器：从 logits 中按 temperature 采样下一个 token。
    
    使用 Gumbel-max trick：
    - 等价于 argmax(logits/temperature + Gumbel(0,1))
    - 实现为 probs / Exp(1).clamp_min(1e-10) 然后 argmax
    - 每轮的 Exp(1) 随机性保证了采样而非贪心
    
    @torch.compile: 利用 PyTorch 2.0 编译，融合 softmax + 除法 + argmax 为单个 CUDA kernel
    """

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """多序列采样。
        
        Args:
            logits: [num_seqs, vocab_size] rank 0 上的完整 logits
            temperatures: [num_seqs] 每条序列的温度参数
        Returns:
            sample_tokens: [num_seqs] 采样的 token IDs
        """
        # Temperature scaling: logits /= temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # Softmax 转概率
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max trick：
        # argmax(probs / Exp(1)) 等价于从 cat(probs) 中采样
        # Exp(1) = -log(U(0,1))，取最小值 clamp 到 1e-10 防止除零
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
