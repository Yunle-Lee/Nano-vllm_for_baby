import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """安全除法，确保整除"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """线性层的基类：处理 TP 切分和权重加载的公共逻辑。
    
    tp_dim 指示沿哪个维度切分权重：
    - tp_dim=0: 沿输出维度切分（Column Parallel）
    - tp_dim=1: 沿输入维度切分（Row Parallel）
    - tp_dim=None: 不切分（Replicated）
    """

    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()      # 当前 GPU 的 TP rank
        self.tp_size = dist.get_world_size() # TP 总 GPU 数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # 绑定自定义加载器
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def weight_loader(self, param, loaded_weight):
        """子类必须实现自己的权重加载逻辑"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """复制线性层：所有 GPU 持有完全相同的权重（不做 TP 切分）。
    用于：Qwen3 中没有被 TP 切分的小层（当前版本未使用）。
    """

    def __init__(self, input_size, output_size, bias=False):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)      # 直接拷贝完整权重

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行线性层：沿输出维度切分权重。
    
    权重布局：[out/tp_size, in]
    每个 GPU 计算输出的一部分，结果不需要通信。
    
    用于：QKV 投影、Gate/Up 投影
    """

    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)   # 当前 rank 应持有的权重行数
        start_idx = self.tp_rank * shard_size       # 在完整权重中的起始位置
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """每个 GPU 独立计算自己那部分输出，无需通信"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并列并行线性层：将多个线性层合并在一起做列并行。
    
    例如 Qwen3 的 gate_proj + up_proj 合并为 gate_up_proj：
    - gate_proj: [intermediate/tp, hidden]
    - up_proj:   [intermediate/tp, hidden]
    合并后: gate_up_proj = [2*intermediate/tp, hidden]
    
    权重文件是分开的（gate_proj 和 up_proj），通过 loaded_shard_id 区分。
    """

    def __init__(self, input_size, output_sizes: list[int], bias=False):
        self.output_sizes = output_sizes           # 例如 [intermediate, intermediate]
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 计算当前 shard 在合并权重中的偏移
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV 并行线性层：将 Q、K、V 三个投影合并到一起做列并行。
    
    权重布局：[Q_size/tp, hidden] | [K_size/tp, hidden] | [V_size/tp, hidden]
    
    Q_size = total_num_heads * head_dim
    K_size = total_num_kv_heads * head_dim (GQA 下 K 和 V 的 head 数可能少于 Q)
    """

    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads=None, bias=False):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)         # 本 rank 的 Q head 数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)   # 本 rank 的 KV head 数
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """QKV 权重的分片加载。
        loaded_shard_id 取值: "q", "k", "v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行线性层：沿输入维度切分权重。
    
    权重布局：[out, in/tp_size]
    每个 GPU 以各自的输入分段进行矩阵乘法 → 然后 all_reduce 求和。
    
    用于：Attention 的 O 投影、MLP 的 Down 投影
    """

    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:               # bias 不切分，直接拷贝
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """各 GPU 独立计算 → all_reduce 求和。
        bias 只有 rank 0 持有，避免重复加。
        """
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)                 # 跨 GPU 汇总
        return y
