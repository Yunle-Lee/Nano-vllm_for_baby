#!/usr/bin/env python3
"""使用 Graphviz 重新绘制架构流程图和模块依赖图"""

import graphviz

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
OUT = "/mnt/workspace/nano-vllm for baby/charts"

# ============================================================
# 架构流程图 — 用 graphviz 画
# ============================================================
def chart3_architecture():
    dot = graphviz.Digraph(
        "architecture",
        format="png",
        engine="dot",
    )
    dot.attr(rankdir="TB", fontname=FONT, fontsize="14",
             label="nano-vllm 推理流水线架构 / Inference Pipeline Architecture",
             labelloc="t", labeljust="c",
             pad="0.5", nodesep="0.4", ranksep="0.6",
             bgcolor="#FAFAFA")

    # 全局节点样式
    dot.attr("node", shape="box", style="filled,rounded", fontname=FONT,
             fontsize="11", penwidth="1.5")

    # 输入/输出
    dot.node("input", "User Prompt\n用户输入", shape="ellipse", fillcolor="#333333", fontcolor="white", fontsize="12")
    dot.node("output", "Generated Text\n生成文本", shape="ellipse", fillcolor="#333333", fontcolor="white", fontsize="12")

    # Tokenizer
    dot.node("tokenizer", "Tokenizer\nencode(prompt) → token IDs", fillcolor="#607D8B", fontcolor="white")

    # Scheduler 组
    with dot.subgraph(name="cluster_scheduler") as c:
        c.attr(label="Scheduler 调度器", fontname=FONT, fontsize="13",
               style="filled,rounded", fillcolor="#FFF3E0", color="#FF5722", penwidth="2")
        c.node("waiting", "Waiting 队列\n待 prefill", fillcolor="#FFC107", fontcolor="#333333")
        c.node("running", "Running 队列\n待 decode", fillcolor="#4CAF50", fontcolor="white")
        c.node("scheduler", "schedule()\n调度决策", fillcolor="#FF5722", fontcolor="white")
        c.node("block_mgr", "BlockManager\nKV 块管理\nPrefix Caching", fillcolor="#E91E63", fontcolor="white")

    # ModelRunner 组
    with dot.subgraph(name="cluster_runner") as c:
        c.attr(label="ModelRunner 模型执行器", fontname=FONT, fontsize="13",
               style="filled,rounded", fillcolor="#E3F2FD", color="#2196F3", penwidth="2")
        c.node("prefill", "prepare_prefill()\ncu_seqlens + slot_mapping", fillcolor="#00BCD4", fontcolor="white")
        c.node("decode_in", "prepare_decode()\n1 token + block_table", fillcolor="#00BCD4", fontcolor="white")
        c.node("model", "Qwen3ForCausalLM\nTransformer Forward", fillcolor="#2196F3", fontcolor="white")
        c.node("cuda_graph", "CUDA Graph Replay\nbatch 1~512", fillcolor="#9C27B0", fontcolor="white", shape="diamond")
        c.node("sampler", "Sampler\nGumbel-max Trick", fillcolor="#795548", fontcolor="white")

    # Attention
    with dot.subgraph(name="cluster_attn") as c:
        c.attr(label="PagedAttention (flash_attn)", fontname=FONT, fontsize="13",
               style="filled,rounded", fillcolor="#FCE4EC", color="#E91E63", penwidth="2")
        c.node("store_kv", "Triton StoreKV\n写入 KV Cache", fillcolor="#E91E63", fontcolor="white")
        c.node("flash_attn", "flash_attn_varlen\nor flash_attn_with_kvcache", fillcolor="#FF5722", fontcolor="white")

    # Postprocess
    dot.node("postprocess", "Scheduler.postprocess()\nhash_blocks + append_token", fillcolor="#795548", fontcolor="white")

    # 连线
    dot.edge("input", "tokenizer")
    dot.edge("tokenizer", "waiting")
    dot.edge("waiting", "scheduler")
    dot.edge("running", "scheduler")
    dot.edge("scheduler", "block_mgr", style="dashed", color="#E91E63")
    dot.edge("block_mgr", "store_kv", label="block_table", style="dashed")

    dot.edge("scheduler", "prefill", label="is_prefill=True")
    dot.edge("scheduler", "decode_in", label="is_prefill=False")
    dot.edge("prefill", "model")
    dot.edge("decode_in", "cuda_graph")
    dot.edge("cuda_graph", "model")
    dot.edge("model", "store_kv", label="K, V")
    dot.edge("store_kv", "flash_attn")
    dot.edge("flash_attn", "sampler", label="attention output")
    dot.edge("sampler", "postprocess", label="token ids")

    # 循环边
    dot.edge("postprocess", "scheduler", style="bold", color="#FF9800",
             label="while not finished\n(循环至全部完成)", fontcolor="#FF9800", fontsize="11")
    dot.edge("postprocess", "output", label="finished seqs", color="#333333")

    # 图例
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="图例 / Legend", fontname=FONT, fontsize="12",
               style="filled,rounded", fillcolor="white", color="#BDBDBD")
        c.node("l1", "调度 Scheduling", fillcolor="#FF5722", fontcolor="white", fontsize="9", height="0.3")
        c.node("l2", "GPU 执行 Execution", fillcolor="#2196F3", fontcolor="white", fontsize="9", height="0.3")
        c.node("l3", "KV Cache", fillcolor="#E91E63", fontcolor="white", fontsize="9", height="0.3")
        c.node("l4", "优化 Optimization", fillcolor="#9C27B0", fontcolor="white", fontsize="9", height="0.3")

    dot.render(f"{OUT}/chart3_architecture", cleanup=True)
    print("chart3_architecture.png (graphviz)")

# ============================================================
# 模块依赖图 — 用 graphviz 画
# ============================================================
def chart6_dependency():
    dot = graphviz.Digraph(
        "dependencies",
        format="png",
        engine="dot",
    )
    dot.attr(rankdir="TB", fontname=FONT, fontsize="14",
             label="nano-vllm 模块依赖关系 / Module Dependency Graph",
             labelloc="t", labeljust="c",
             pad="0.3", nodesep="0.3", ranksep="0.5",
             bgcolor="#FAFAFA")
    dot.attr("node", shape="box", style="filled,rounded", fontname=FONT,
             fontsize="9", penwidth="1.2", margin="0.1,0.06")
    dot.attr("edge", color="#78909C", arrowsize="0.6", fontname=FONT, fontsize="7")

    # 按层排列
    with dot.subgraph() as s:
        s.attr(rank="source")
        s.node("llm", "llm.py\n(用户入口)", fillcolor="#333333", fontcolor="white", fontsize="10")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("llm_engine", "llm_engine\n(主循环)", fillcolor="#FF5722", fontcolor="white")
        s.node("config", "config\n(配置)", fillcolor="#4CAF50", fontcolor="white")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("scheduler", "scheduler\n(调度器)", fillcolor="#FF9800", fontcolor="white")
        s.node("model_runner", "model_runner\n(模型执行)", fillcolor="#2196F3", fontcolor="white")
        s.node("sampling_params", "sampling_params\n(采样参数)", fillcolor="#795548", fontcolor="white")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("block_manager", "block_manager\n(块管理+Pfx缓存)", fillcolor="#E91E63", fontcolor="white")
        s.node("sequence", "sequence\n(序列状态)", fillcolor="#607D8B", fontcolor="white")
        s.node("context", "context\n(全局上下文)", fillcolor="#FF9800", fontcolor="white", shape="diamond")
        s.node("loader", "loader\n(权重加载)", fillcolor="#607D8B", fontcolor="white")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("qwen3", "qwen3.py\n(模型定义)", fillcolor="#9C27B0", fontcolor="white")
        s.node("sampler", "sampler\n(采样器)", fillcolor="#795548", fontcolor="white")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("attention", "attention\n(PagedAttention)", fillcolor="#00BCD4", fontcolor="white")
        s.node("linear", "linear\n(TP 线性层)", fillcolor="#3F51B5", fontcolor="white")
        s.node("embed_head", "embed_head\n(词表并行)", fillcolor="#00BCD4", fontcolor="white")
        s.node("rotary_emb", "rotary_emb\n(RoPE)", fillcolor="#FFC107", fontcolor="#333333")
        s.node("layernorm", "layernorm\n(RMSNorm)", fillcolor="#8BC34A", fontcolor="white")
        s.node("activation", "activation\n(SwiGLU)", fillcolor="#8BC34A", fontcolor="white")

    # 连线
    edges = [
        ("llm", "llm_engine"),
        ("llm_engine", "scheduler"), ("llm_engine", "model_runner"), ("llm_engine", "config"),
        ("llm_engine", "sequence"), ("llm_engine", "sampling_params"),
        ("scheduler", "block_manager"), ("scheduler", "config"), ("scheduler", "sequence"),
        ("block_manager", "sequence"),
        ("model_runner", "qwen3"), ("model_runner", "sampler"), ("model_runner", "config"),
        ("model_runner", "sequence"), ("model_runner", "context"), ("model_runner", "loader"),
        ("qwen3", "attention"), ("qwen3", "linear"), ("qwen3", "embed_head"),
        ("qwen3", "rotary_emb"), ("qwen3", "layernorm"), ("qwen3", "activation"),
        ("attention", "context"), ("embed_head", "context"),
    ]
    for s, d in edges:
        dot.edge(s, d)

    # 图例
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="图例 / Legend", fontname=FONT, fontsize="11",
               style="filled,rounded", fillcolor="white", color="#BDBDBD", rank="sink")
        c.node("L1", "用户入口 / Entry", fillcolor="#333333", fontcolor="white", fontsize="8", height="0.25")
        c.node("L2", "engine/ 引擎", fillcolor="#FF5722", fontcolor="white", fontsize="8", height="0.25")
        c.node("L3", "models/ 模型", fillcolor="#9C27B0", fontcolor="white", fontsize="8", height="0.25")
        c.node("L4", "layers/ 算子", fillcolor="#00BCD4", fontcolor="white", fontsize="8", height="0.25")
        c.node("L5", "utils/ 工具", fillcolor="#FF9800", fontcolor="white", fontsize="8", height="0.25")
        c.node("L6", "配置 & 参数", fillcolor="#4CAF50", fontcolor="white", fontsize="8", height="0.25")

    dot.render(f"{OUT}/chart6_dependency", cleanup=True)
    print("chart6_dependency.png (graphviz)")


if __name__ == "__main__":
    chart3_architecture()
    chart6_dependency()
    print(f"\nCharts regenerated in {OUT}/")
