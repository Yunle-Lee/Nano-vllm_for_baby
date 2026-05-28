from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """全局上下文：在 prefill/decode 之间隐式传递 Attention 计算所需的元信息。
    
    使用全局变量模式而非显式函数参数传递，避免在每个函数签名中追加大量参数。
    
    字段在不同阶段的使用：
    ┌──────────────────┬─────────────────────┬─────────────────────┐
    │ 字段             │ Prefill             │ Decode              │
    ├──────────────────┼─────────────────────┼─────────────────────┤
    │ is_prefill       │ True                │ False               │
    │ cu_seqlens_q     │ Q 累计长度 [N+1]    │ None                │
    │ cu_seqlens_k     │ KV 累计长度 [N+1]   │ None                │
    │ max_seqlen_q     │ 最大 Q 序列长度     │ 0                   │
    │ max_seqlen_k     │ 最大 KV 序列长度    │ 0                   │
    │ slot_mapping     │ 写入位置 [total_tok]│ 写入位置 [N]        │
    │ context_lens     │ None                │ 各序列 KV 长度 [N]  │
    │ block_tables     │ 前缀缓存时的页表     │ 页表 [N, max_blocks]│
    └──────────────────┴─────────────────────┴─────────────────────┘
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


# 全局单例 Context 实例
_CONTEXT = Context()


def get_context():
    """获取全局上下文（在 Attention 层中调用）"""
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None):
    """设置全局上下文（在 prepare_prefill/decode 中调用）"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables)


def reset_context():
    """重置全局上下文（每次 step 结束后调用）"""
    global _CONTEXT
    _CONTEXT = Context()
