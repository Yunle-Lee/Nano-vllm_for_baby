import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """全局配置中心，由用户传入参数和自动检测的模型参数组成。
    
    每个字段都有合理的默认值，最小化用户必填参数。
    """
    model: str                           # 本地模型路径（唯一必填参数）
    max_num_batched_tokens: int = 16384  # 单次 prefill 最多处理的 token 数
    max_num_seqs: int = 512              # 同时处理的序列数上限
    max_model_len: int = 4096            # 模型最大序列长度（会被 hf_config 约束上限）
    gpu_memory_utilization: float = 0.9  # GPU 显存利用率（预留 10% 用于临时开销）
    tensor_parallel_size: int = 1        # 张量并行 GPU 数量（1~8）
    enforce_eager: bool = False          # True=禁用 CUDA Graph，用于调试
    hf_config: AutoConfig | None = None  # HuggingFace 模型配置（__post_init__ 中自动加载）
    eos: int = -1                        # EOS token id（从 tokenizer 自动获取）
    kvcache_block_size: int = 256        # 每个 KV cache block 的 token 数（必须是 256 的倍数）
    num_kvcache_blocks: int = -1         # KV cache block 总数（allocate_kv_cache 中动态计算）

    def __post_init__(self):
        """初始化后自动校验和加载 HuggingFace 配置"""
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0  # block 大小必须是 256 的倍数，确保对齐
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
