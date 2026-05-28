from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """对外暴露的用户 API 类，直接继承 LLMEngine。
    
    设计说明：
    这里 LLM 是一个空子类，完全继承 LLMEngine 的功能。
    这种设计是为了保持 vLLM 兼容的 API 风格。
    用户只需 from nanovllm import LLM 就能使用。
    """
    pass
