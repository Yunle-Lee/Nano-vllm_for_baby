import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU 激活函数：SiLU(gate) * up。

    输入 x 是 gate 和 up 拼接的结果（shape 最后一维是 2*intermediate_size）：
    1. chunk(2, -1)：沿最后一维切分为两半
       - x（前半）= gate 的输出
       - y（后半）= up 的输出
    2. SiLU(x) * y：gate 经过 SiLU 激活后与 up 元素乘

    SiLU(x) = x * sigmoid(x)

    @torch.compile: 将 chunk + silu + mul 融合为一个 CUDA kernel
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)      # 切分：x = gate, y = up
        return F.silu(x) * y       # SwiGLU
