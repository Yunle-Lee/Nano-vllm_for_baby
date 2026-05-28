from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """采样参数：控制模型如何从 logits 中选择下一个 token。
    
    当前实现非常简洁，只支持 temperature-based 随机采样，
    不支持 greedy sampling、top-k、top-p 等常见策略。
    """
    temperature: float = 1.0    # 温度参数，控制输出的随机性（必须 > 1e-10）
    max_tokens: int = 64        # 最多生成的 token 数
    ignore_eos: bool = False    # 是否忽略 EOS 继续生成

    def __post_init__(self):
        """temperature 必须大于 1e-10，不允许 greedy sampling"""
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
