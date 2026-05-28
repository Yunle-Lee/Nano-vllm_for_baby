import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """推理引擎主控制器：连接用户 API、调度器和模型执行器。
    
    职责：
    1. 启动 Tensor Parallelism 工作进程（rank 1~N）
    2. 管理主推理循环（generate 方法中的 while not is_finished 循环）
    3. 吞吐量监控（Prefill / Decode tok/s 实时显示）
    4. 请求生命周期管理（tokenize → add → schedule → run → postprocess → decode）
    """

    def __init__(self, model, **kwargs):
        """初始化引擎。
        
        Args:
            model: 本地模型路径
            **kwargs: 传递给 Config 的额外参数（如 enforce_eager, tensor_parallel_size）
        """
        # 从 kwargs 中提取 Config 支持的字段，其余忽略
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 设置类变量：所有 Sequence 共享同一个 block_size
        Sequence.block_size = config.kvcache_block_size

        # ========== 启动 Tensor Parallelism 工作进程 ==========
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")                  # 使用 spawn 方式创建子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()                        # 每个 worker 一个 Event 用于同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 主进程自己就是 rank 0（接收所有 worker 的 events 列表）
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id       # 记录 EOS token id

        self.scheduler = Scheduler(config)

        # 注册退出清理函数（确保异常退出时也释放资源）
        atexit.register(self.exit)

    def exit(self):
        """清理所有资源"""
        self.model_runner.call("exit")                 # 通知所有 worker 退出
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """接收一条用户请求，tokenize 后加入调度器"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)     # 字符串 → token IDs
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """执行一个推理 step：schedule → run → postprocess。
        
        Returns:
            (outputs, num_tokens):
            - outputs: [(seq_id, completion_token_ids), ...] 本轮完成的序列
            - num_tokens > 0: prefill 处理的 token 数
            - num_tokens < 0: decode 处理的序列数（取绝对值 = 序列数）
        """
        seqs, is_prefill = self.scheduler.schedule()

        # 统计 token 数（用于吞吐量计算）
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """所有请求是否都已完成"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """主推理循环：接受一批 prompt，返回生成的文本。
        
        Args:
            prompts: 输入文本列表（字符串或预编码的 token ID 列表）
            sampling_params: 采样参数（单个或每个 prompt 一个）
            use_tqdm: 是否显示进度条
        
        Returns:
            outputs: [{"text": str, "token_ids": list[int]}, ...]
        """
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 如果 sampling_params 是单个值，给每个 prompt 复制一份
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 将所有请求加入等待队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # 主循环：不断 step 直到所有请求完成
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            # 计算吞吐量
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)

            # 更新进度条
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 收集完成的输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()

        # 按 seq_id 排序输出（保持和输入顺序一致）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
                   for token_ids in outputs]
        return outputs
