from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列生命周期状态机：
    WAITING  → 等待首次 prefill（在 waiting 队列中）
    RUNNING  → prefill 完成，正在逐 token 生成（在 running 队列中）
    FINISHED → 生成完成（已从队列中移除）
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """一条推理请求的完整状态。
    
    这是整个推理引擎中最核心的数据结构，贯穿调度、KV cache 管理和模型执行。
    每条 Sequence 唯一对应一个用户请求（prompt + 生成结果）。
    
    核心概念：
    - token_ids: 完整 token 序列（prompt + 已生成的 completion）
    - block_table: PagedAttention 的页表，存储该序列使用的物理 KV cache 块编号
                    例如 [3, 7, 12] 表示该序列的 KV 存储在物理块 3、7、12 中
    - num_cached_tokens: 已写入 KV cache 的 token 数（prefix caching 命中后可以 > 0）
    - num_scheduled_tokens: 本轮 step 要处理的 token 数
    """
    # 类变量：KV cache 的块大小，由 Config 注入（默认 256）
    # 设置为类变量方便在没有实例化 Config 的地方也能获取
    block_size = 256
    # 全局自增计数器，为每条序列分配唯一 ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)       # 全局唯一序列 ID
        self.status = SequenceStatus.WAITING        # 初始状态：等待首次 prefill
        self.token_ids = copy(token_ids)            # 完整 token 序列（浅拷贝）
        self.last_token = token_ids[-1]             # 最后一个 token（decode 时只需这个）
        self.num_tokens = len(self.token_ids)       # 当前总 token 数（prompt + 已生成）
        self.num_prompt_tokens = len(token_ids)     # 原始 prompt 的 token 数（不变）
        self.num_cached_tokens = 0                  # 已存入 KV cache 的 token 数
        self.num_scheduled_tokens = 0               # 本轮被调度处理的 token 数
        self.is_prefill = True                      # 是否处于 prefill 阶段
        self.block_table = []                       # 页表：[物理块ID_0, 物理块ID_1, ...]
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回序列的 token 总数"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持切片访问 token_ids，如 seq[start:end]"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 completion token 数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """只返回 prompt 部分（不含已生成的 completion）"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """只返回已生成的 completion 部分"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """该序列需要的 KV cache 块总数（向上取整）"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中存储的 token 数（最后一个块通常不完整）"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第 i 个块应存储的 token 序列（用于计算前缀缓存哈希）"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """解码阶段生成一个新 token 后调用，将 token 追加到序列末尾"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """Tensor Parallelism 多进程间序列化：只传必要字段。
        
        prefill 阶段传输完整 token_ids（较多个 token 但只发生一次），
        decode 阶段只传 last_token（1 个 token，高频传输）。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """反序列化：根据收到的状态重建 Sequence"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, \
            self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):      # prefill 阶段收到了完整 token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:                                 # decode 阶段只收到了 last_token
            self.token_ids = []
            self.last_token = last_state
