from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """调度器：推理引擎的「大脑」，决定每轮 step 应该处理哪些序列。
    
    核心职责：
    1. 维护 waiting / running 两个队列
    2. 每轮 schedule() 决定做 prefill 还是 decode，以及选择哪些序列
    3. postprocess() 处理模型输出，更新序列状态
    
    状态流转：
        新请求 → waiting ──(prefill完成)──→ running ──(EOS/max_tokens)──→ FINISHED
                           ↑                    │
                           └──(preempt 抢占)────┘
    
    关键设计决策：
    - prefill 和 decode 不同时进行（简化实现）
    - 先 prefill 后 decode：优先让新序列完成 prefill
    - FIFO 调度（waiting 队列先进先出，running 队列轮转）
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos                         # EOS token id
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()       # 等待首次 prefill 的序列
        self.running: deque[Sequence] = deque()       # prefill 完成，正在 decode 的序列

    def is_finished(self):
        """引擎是否完成所有任务（两个队列都为空）"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """新增一条请求到等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """核心调度函数，每轮 step 调用一次。

        返回值:
            (scheduled_seqs, is_prefill)
            is_prefill=True  → 这批序列做 prefill
            is_prefill=False → 这批序列做 decode
        
        调度顺序：先尝试 prefill，没有可 prefill 的序列才做 decode。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ==================== 第一阶段：Prefill ====================
        # 从 waiting 队列中取出序列，给它们分配 KV cache 并做首次前向传播
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]                      # FIFO：取队首
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break                                  # batch token 配额用尽

            if not seq.block_table:                    # 首次分配 KV cache
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break                              # 显存不足，暂停 prefill
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 当前序列之前被预分配过（chunked prefill 的后续轮次），
                # 但 block_table 被清空了（被抢占过），需要重新判断
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Chunked Prefill：如果当前序列的剩余 token 数超过配额，
            # 且已经有其他序列在 batch 中，则本轮不打断当前序列（等下一轮）
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 本轮实际处理的 token 数：不能超过剩余配额
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 当所有 token 都被预处理完，序列正式进入 RUNNING 状态
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True               # 返回 prefill batch

        # ==================== 第二阶段：Decode ====================
        # 从 running 队列中取出序列，每条只处理 1 个 token
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()              # 取出队首（FIFO轮转）

            # 检查是否有足够空闲块追加（decode 阶段可能需要新块）
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())  # 抢占最后的序列，腾出显存
                else:
                    self.preempt(seq)                 # 只有自己，只能抢占自己
                    break
            else:
                seq.num_scheduled_tokens = 1          # decode 每次只处理 1 token
                seq.is_prefill = False
                self.block_manager.may_append(seq)    # 如果 token 数到达块边界，追加新块
                scheduled_seqs.append(seq)

        assert scheduled_seqs                         # 必须有序列可处理
        # 轮转调度：处理完的序列放回队尾，下次会处理其他序列
        # reversed + extendleft 保证相对顺序不变，只是整体左移
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False                  # 返回 decode batch

    def preempt(self, seq: Sequence):
        """抢占机制：显存不足时，将一条 running 序列退回 waiting 状态。
        
        释放该序列的所有 KV cache 块，下次重新 prefill。
        appendleft 确保被抢占的序列有最高优先级，不会饿死。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True                         # 标记需要重新 prefill
        self.block_manager.deallocate(seq)            # 释放 KV cache 块
        self.waiting.appendleft(seq)                  # 放回 waiting 队首（优先处理）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """模型执行完成后的后处理。
        
        1. 将新填满的块写入前缀缓存（hash_blocks）
        2. 更新 cached_tokens 计数
        3. 如果 prefill 还没完成（chunked prefill），跳过后续步骤
        4. 追加新 token，判断是否满足终止条件
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)       # 写入前缀缓存
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # Chunked Prefill：还没完成，不采样 token
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)

            # 终止判断：
            # 1) 生成了 EOS token（且未忽略 EOS）
            # 2) 生成的 completion token 数达到 max_tokens 上限
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)    # 释放 KV cache
                self.running.remove(seq)              # 从运行队列中移除
