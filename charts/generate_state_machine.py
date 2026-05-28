#!/usr/bin/env python3
"""用 Graphviz 重新绘制 Sequence 状态机图"""

import graphviz

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
OUT = "/mnt/workspace/nano-vllm for baby/charts"

dot = graphviz.Digraph("state_machine", format="png", engine="dot")
dot.attr(rankdir="LR", fontname=FONT, fontsize="14",
         label="Sequence 状态机 / Sequence State Machine\nWAITING → RUNNING → FINISHED with Preemption",
         labelloc="t", labeljust="c",
         bgcolor="#FAFAFA",
         pad="0.5", nodesep="0.6", ranksep="0.3")

dot.attr("node", shape="box", style="filled,rounded", fontname=FONT,
         fontsize="12", penwidth="2", margin="0.25,0.2")
dot.attr("edge", fontname=FONT, fontsize="11", penwidth="2", arrowsize="1.0")

# 三个主状态
dot.node("WAITING", "WAITING\n等待首次 prefill\nAwaiting first prefill",
         fillcolor="#FF9800", fontcolor="white")
dot.node("RUNNING", "RUNNING\n逐 token 生成中\nGenerating tokens",
         fillcolor="#4CAF50", fontcolor="white")
dot.node("FINISHED", "FINISHED\n生成完成\nGeneration complete",
         fillcolor="#9E9E9E", fontcolor="white")

# 转换
dot.edge("WAITING", "RUNNING",
         label="prefill 完成 / prefill done\n→ scheduler 移入 running 队列",
         color="#4CAF50")
dot.edge("RUNNING", "FINISHED",
         label="EOS token\nor max_tokens 达到上限",
         color="#9E9E9E")

# Preempt 抢占
dot.edge("RUNNING", "WAITING",
         label="Preemption 抢占\n显存不足时释放 KV Cache\n放回 waiting 队列队首",
         color="#FF5722", style="dashed", constraint="false")

# add_request
dot.node("start", "add_request()\n新增请求", shape="ellipse",
         fillcolor="#333333", fontcolor="white", fontsize="12")
dot.edge("start", "WAITING", color="#333333")

# 图例
with dot.subgraph(name="cluster_legend") as c:
    c.attr(label="图例 / Legend", fontname=FONT, fontsize="11",
           style="filled,rounded", fillcolor="white", color="#BDBDBD", rank="sink")
    c.node("L1", "", fillcolor="#FF9800", fontcolor="white", shape="box", width="0.4", height="0.2")
    c.node("L2", "", fillcolor="#4CAF50", fontcolor="white", shape="box", width="0.4", height="0.2")
    c.node("L3", "", fillcolor="#9E9E9E", fontcolor="white", shape="box", width="0.4", height="0.2")

dot.render(f"{OUT}/chart8_state_machine", cleanup=True)
print("chart8_state_machine.png (graphviz)")
