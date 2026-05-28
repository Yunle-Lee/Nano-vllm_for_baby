import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """模型执行器：直接与 GPU 交互，负责模型加载、前向推理和 Tensor Parallelism 通信。
    
    这是整个引擎中最复杂的模块（~260 行），集成了：
    1. 模型加载和热机（warmup）
    2. KV Cache 显存分配（根据剩余显存动态计算可分配的 block 数）
    3. Prefill / Decode 输入准备（构造 input_ids, positions, slot_mapping 等）
    4. CUDA Graph 捕获（decode 阶段加速）
    5. Tensor Parallelism 多进程通信（SharedMemory RPC）
    6. 采样（Sampler）
    
    TP 架构：
    - rank 0 在主进程中运行，负责任务分发和采样
    - rank 1~N 在独立子进程中运行，只做前向计算
    - 通信方式：NCCL（GPU 间 all_reduce/gather）+ SharedMemory（CPU 间 RPC）
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager   # 是否禁用 CUDA Graph
        self.world_size = config.tensor_parallel_size
        self.rank = rank                            # 当前进程的 TP rank（0 = 主进程）
        self.event = event                          # 用于跨进程同步的 multiprocessing.Event

        # 初始化 NCCL 进程组，用于 GPU 间的 all_reduce/gather 通信
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # 保存原有 dtype 和设备，模型需要运行在 CUDA 上用 bf16
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)     # 设置模型默认 dtype（如 bfloat16）
        torch.set_default_device("cuda")             # tensor 默认在 GPU 上创建

        # 构建模型 → 加载权重 → 创建采样器 → 热机
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()                          # 用伪数据跑一次，触发 CUDA kernel 编译

        # 根据 GPU 剩余显存动态分配 KV Cache
        self.allocate_kv_cache()

        # 如果不强制 eager，为 decode 阶段捕获 CUDA Graph
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复默认设置，避免影响外部代码
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ========== Tensor Parallelism 多进程通信设置 ==========
        if self.world_size > 1:
            if rank == 0:
                # 主进程创建 SharedMemory（1MB），供所有 worker 读取
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()                       # 等待 worker 就绪
            else:
                # Worker 进程等待主进程创建好 SharedMemory 后连接
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()                          # 进入事件循环，等待主进程指令

    def exit(self):
        """清理资源"""
        if self.world_size > 1:
            self.shm.close()                         # 关闭共享内存
            dist.barrier()                           # 同步确保所有进程都已关闭
            if self.rank == 0:
                self.shm.unlink()                    # 只有创建者才能 unlink
        if not self.enforce_eager:
            del self.graphs, self.graph_pool         # 释放 CUDA Graph 显存
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Worker 进程的事件循环：不断读取 SharedMemory 指令并执行"""
        while True:
            method_name, args = self.read_shm()      # 阻塞等待主进程指令
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """从 SharedMemory 中读取主进程发来的 RPC 调用"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()                            # 阻塞等待，直到主进程写入数据
        n = int.from_bytes(self.shm.buf[0:4], "little")   # 前 4 字节 = 数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化方法名和参数
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """向 SharedMemory 写入 RPC 调用（主进程调用）"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])    # 序列化
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入长度
        self.shm.buf[4:n+4] = data                   # 写入数据
        for event in self.event:
            event.set()                               # 通知所有 worker 进程

    def call(self, method_name, *args):
        """RPC 分发：如果主进程且 TP>1，先通过 SharedMemory 广播到 worker"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """模型热机：用伪数据执行一次 prefill 和 decode，触发 CUDA kernel JIT 编译。
        
        这确保后续实际推理时不会因为 kernel 编译而产生首次延迟。
        同时重置显存统计，保证后续 allocate_kv_cache 的显存计算准确。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]   # 伪序列（全0）
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)                         # 一次 prefill + forward
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据 GPU 剩余显存动态计算并分配 KV Cache。
        
        显存分配公式：
            可分配 = 总显存 × gpu_memory_utilization - 模型权重 - 临时分配 + 当前 PyTorch 占用
        
        说明：
        - peak: 模型加载完成后 PyTorch 分配器的峰值内存
        - current: 当前 PyTorch 分配器占用的内存
        - peak - current: 可以在分配器内部重用的空间（warmup 释放的临时内存）
        
        KV Cache 形状：[2, layers, num_blocks, block_size, kv_heads, head_dim]
        - 第 0 维：2 表示 K 和 V 分开存储
        - 注入到每层 Attention 的 k_cache / v_cache 中
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()      # GPU 可用/总显存
        used = total - free                          # 当前已用显存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]     # 峰值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 当前

        # 计算单个 KV cache block 的字节数（K + V = 2x）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size   # TP 切分 KV heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # 计算可分配的 block 数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 创建 KV cache tensor 并映射到每层 Attention
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
                                     self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]    # K cache 的第 layer_id 层
                module.v_cache = self.kv_cache[1, layer_id]    # V cache 的第 layer_id 层
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """构造 block_tables tensor：每条序列的页表对齐到相同长度（不足补 -1）。
        
        返回: tensor [num_seqs, max_blocks]，类型 int32
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prefill 阶段输入准备。
        
        flash_attn_varlen_func 需要的参数：
        - input_ids: 所有序列本轮要处理的 token 拼接在一起 [total_tokens]
        - positions: 每个 token 在序列中的位置 [total_tokens]
        - cu_seqlens_q/k: query 和 key 的累积序列长度（varlen 格式）[num_seqs+1]
                           cu_seqlens[i+1] - cu_seqlens[i] = 第 i 条序列的长度
        - slot_mapping: 每个 token 应写入 KV cache 的位置 [total_tokens]
                         slot_mapping[i] = block_table[block_i] * block_size + offset
        - block_tables: 前缀缓存命中时需要，用于 flash_attn 做跨块索引
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]           # 积累 query 长度
        cu_seqlens_k = [0]           # 积累 key 长度（含缓存前缀，可能 > query 长度）
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            start = seq.num_cached_tokens              # 已缓存 token 数（前缀缓存命中时 > 0）
            seqlen_q = seq.num_scheduled_tokens        # 本轮 prefill 的 token 数
            end = start + seqlen_q
            seqlen_k = end                             # KV 包含缓存和下文的全部长度

            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:                    # warmup 阶段无 block_table
                continue

            # 构造 slot_mapping：每个 token 应该写入 KV cache 的哪个位置
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size  # 第一个块可能从非 0 位置开始
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        # 前缀缓存检测：cu_seqlens_k > cu_seqlens_q 说明有 PV 长度 > Q 长度
        # 即有些块的 KV 已经在缓存中，需要通过 block_tables 索引
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 异步传输到 GPU（pin_memory + non_blocking）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 写入全局上下文，供 Attention 层读取
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Decode 阶段输入准备。
        
        与 prefill 不同：
        - 每条序列只取 1 个 token（last_token）
        - 使用 flash_attn_with_kvcache 接口
        - 需要 context_lens（每条序列的 KV 长度）而不是 cu_seqlens
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)           # 只取最后一个 token
            positions.append(len(seq) - 1)             # 最后一个 token 的位置
            context_lens.append(len(seq))              # 完整 KV 长度

            # slot_mapping：新生成的 token 应该写入 KV cache 的哪个位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # 异步传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # 写入全局上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数：每条序列的 temperature"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """模型前向推理。prefill 走 eager，decode 优先走 CUDA Graph。
        
        CUDA Graph 使用条件：
        - decode 阶段（非 prefill）
        - enforce_eager = False
        - batch_size ≤ 512（大于 512 走 eager，因为 graph 只捕获到 512）
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager 模式：直接调用模型
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph Replay 模式
            bs = input_ids.size(0)
            context = get_context()
            # 找到 >= bs 的最小 graph batch size
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # 将本轮数据写入预分配的 buffer
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放 CUDA Graph，零 CPU launch overhead
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """完整的推理 step：准备输入 → 前向 → 采样 → 返回 token_ids。
        
        只有 rank 0 执行采样（因为只有 rank 0 有完整的 logits）。
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """CUDA Graph 捕获：为 decode 阶段捕获多种 batch size 的全模型前向计算图。
        
        CUDA Graph 原理：
        1. warmup：正常执行一次，触发所有 CUDA kernel 编译
        2. capture：用 torch.cuda.graph 录制所有 kernel 启动
        3. replay：graph.replay() 一次性提交所有 kernel，省去 CPU launch overhead
        
        捕获的 batch sizes：[1, 2, 4, 8, 16, 32, ..., max_bs]
        从大到小捕获：先捕获大 batch 确定 mempool，小 batch 复用同一 pool。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)   # 最多捕获到 512
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 预分配最大 batch size 的 buffer
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None                         # 第一个 graph 创建 pool，后续复用

        for bs in reversed(self.graph_bs):             # 从大到小
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs],
                        context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()         # 缓存第一个 graph 的 pool
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存所有 buffer 引用，replay 时会修改它们的内容
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
