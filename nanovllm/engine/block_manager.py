from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """KV Cache 的物理块。
    
    每个 Block 存储固定数量（block_size）token 的 Key/Value。
    GPU 显存中有一块巨大的连续 tensor（k_cache/v_cache），
    通过 block_table 将逻辑顺序映射到物理块。
    
    Prefix Caching：
    - hash: 该 block 内 token_ids 的链式 xxhash 值，用于前缀匹配
    - token_ids: 块内 token 序列，用于二次验证哈希碰撞
    - ref_count: 引用计数，多序列共享同一块时 >1
    """

    def __init__(self, block_id):
        self.block_id = block_id      # 物理块在 KV cache tensor 中的索引
        self.ref_count = 0            # 引用计数
        self.hash = -1                # 前缀缓存哈希值（-1 = 未哈希）
        self.token_ids = []           # 块内 token 序列（用于精确验证哈希碰撞）

    def update(self, hash: int, token_ids: list[int]):
        """写入块：设置哈希值和 token 序列"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态（回收后再分配时调用）"""
        self.ref_count = 1            # 初始引用计数 = 1（被当前序列持有）
        self.hash = -1                # 清除哈希
        self.token_ids = []           # 清除 token 序列


class BlockManager:
    """KV Cache 物理块管理器 — PagedAttention 和 Prefix Caching 的实现核心。
    
    职责：
    1. 块分配/回收：管理空闲块队列，分配新块给序列，回收用完的块
    2. 前缀缓存：基于 xxhash 链式哈希识别请求间的公共前缀
                多序列可共享同一物理块（通过 ref_count 引用计数管理）
    
    数据结构：
    - free_block_ids: 空闲块 ID 队列（FIFO）
    - used_block_ids: 已分配块 ID 集合（用于快速查找）
    - hash_to_block_id: {哈希值 → block_id} 前缀缓存索引
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 初始全部空闲
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """链式哈希计算。
        
        关键设计：prefix 参数是前一个块的哈希值，这样：
        Hash(Block_i) = xxhash(Block_i_tokens, prev_hash=Hash(Block_{i-1}))
        
        好处：如果两序列的前 N 个块完全相同，Hash(Block_0) 到 Hash(Block_{N-1})
        也会完全相同，可以直接匹配整个前缀。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))    # 将前一块的哈希混入
        h.update(np.array(token_ids).tobytes())       # 混入本块的 token 序列
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲池中分配一个新块"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 如果这个块之前有前缀缓存记录，需要从哈希表中移除
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """归还一个块到空闲池"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """检查是否可以分配显存给新序列，同时检测前缀缓存命中。
        
        返回:
            -1 → 显存不足，无法分配
            ≥0 → 可以分配，返回值 = 前缀缓存命中的块数
        
        算法：
            逐块计算 token 哈希，与已有块的哈希表匹配。
            找到最长连续匹配前缀，未被其他序列使用的匹配块不需要申请新显存。
        """
        h = -1                              # 链式哈希累积值
        num_cached_blocks = 0               # 哈希命中的块数
        num_new_blocks = seq.num_blocks     # 需要新分配的块数（初始 = 总块数）
        for i in range(seq.num_blocks - 1): # 最后一个不完整块不参与前缀缓存
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break                       # 匹配失败，停止查找
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1         # 块已被使用（可直接共享），不需要新分配
        if len(self.free_block_ids) < num_new_blocks:
            return -1                       # 空闲块不够
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为新序列分配物理块。num_cached_blocks 是 can_allocate 返回的前缀缓存命中数。
        
        分配策略：
        - 前 num_cached_blocks 个块：复用已有块（ref_count++），或从空闲池取
        - 剩余块：从空闲池分配新块
        """
        assert not seq.block_table           # 确保序列尚未分配过
        h = -1
        for i in range(num_cached_blocks):   # 前缀缓存命中的块
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1         # 已在使用的块，增加引用计数即可
            else:
                block.ref_count = 1          # 在哈希表中但未使用的块，标记为使用
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())  # 分配全新块
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有块（ref_count--）。
        
        逆序处理 block_table 确保正确的引用计数递减顺序。
        只有当块的所有引用都释放后（ref_count == 0），才归还空闲池。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查 decode 阶段是否能为序列追加一个新块。
        
        decode 阶段每生成 1 个 token，只有当 token 数恰好到达块边界时才需要新块。
        例如 block_size=256，序列长度从 256→257 时，最后一个块刚好满，需要新块。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 阶段：如果序列长度刚好到达块边界，分配一个新物理块"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """将新填充的物理块写入前缀缓存。
        
        只哈希本轮新填满的块（从 start 到 end-1）。
        链式哈希：每个块的哈希基于前一个块的哈希值计算，确保前缀连续性。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return                          # 本轮没有填满任何新块
        # 获取前一个块的哈希作为起点（start > 0 时从 block_table[start-1] 获取）
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)      # 更新块的哈希和 token 记录
            # 注册到哈希表（若同一值已被其他块占用则覆盖）
            self.hash_to_block_id[h] = block.block_id
