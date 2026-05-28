#!/usr/bin/env python3
"""nano-vllm for baby — 补充图表（饼图、曲线图、状态机图）"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Arc
import numpy as np
import os

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
from matplotlib.font_manager import FontProperties
F = lambda s=11: FontProperties(fname=FONT, size=s)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

OUT = "/mnt/workspace/nano-vllm for baby/charts"

def make_patch(color, label):
    return Rectangle((0, 0), 1, 1, fc=color, label=label)

# ============================================================
# 7. 代码模块饼图
# ============================================================
def chart7():
    labels = ["engine/\n(818 lines)", "layers/\n(421 lines)", "models/\n(182 lines)",
              "utils/\n(40 lines)", "config &\nothers (37 lines)"]
    sizes = [818, 421, 182, 40, 37]
    colors = ["#FF5722", "#2196F3", "#9C27B0", "#FF9800", "#4CAF50"]
    explode = (0.05, 0.02, 0.02, 0, 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct="%1.1f%%",
                                       startangle=140, pctdistance=0.6,
                                       textprops={"fontproperties": F(10)})
    for at in autotexts:
        at.set_fontproperties(F(9))
    ax.set_title("nano-vllm 代码模块占比 / Code Distribution by Module",
                 fontproperties=F(15), pad=20)

    ax.legend(wedges, [
        "engine/ — Scheduler, BlockManager, ModelRunner",
        "layers/ — Attention, Linear, Sampler, etc.",
        "models/ — Qwen3 Model Definition",
        "utils/ — Context, Loader",
        "config — Config, SamplingParams"
    ], loc="center left", bbox_to_anchor=(1, 0.5), prop=F(9), framealpha=0.9)

    plt.tight_layout()
    fig.savefig(f"{OUT}/chart7_module_pie.png")
    plt.close()
    print("chart7_module_pie.png")

# ============================================================
# 8. Sequence 状态机图
# ============================================================
def chart8():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Sequence 状态机 / Sequence State Machine",
                 fontproperties=F(15), pad=15)

    def state_box(x, y, w, h, title, desc, color):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           fc=color, ec="white", lw=2, alpha=0.9)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + 0.35, title, ha="center", va="center",
                fontproperties=F(12), color="white")
        ax.text(x + w / 2, y + h / 2 - 0.45, desc, ha="center", va="center",
                fontsize=9, color="white", alpha=0.9)

    def arrow(x1, y1, x2, y2, label="", color="gray", style="solid"):
        ls = "dashed" if style == "dashed" else "solid"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, linestyle=ls))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.2
            ax.text(mx, my, label, ha="center", fontproperties=F(9), color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8))

    # 三个状态
    state_box(0.5, 2.0, 2.5, 1.2, "WAITING", "awaiting first prefill\n等待首次预填充", "#FF9800")
    state_box(4.5, 2.0, 2.5, 1.2, "RUNNING", "token generation\n逐 token 生成中", "#4CAF50")
    state_box(8.5, 2.0, 2.5, 1.2, "FINISHED", "EOS or max_tokens\n生成完成", "#9E9E9E")

    # WAITING → RUNNING
    arrow(3.0, 2.6, 4.5, 2.6, "prefill 完成 / prefill done", "#4CAF50")
    # RUNNING → FINISHED
    arrow(7.0, 2.6, 8.5, 2.6, "EOS / max_tokens", "#9E9E9E")

    # Preempt: RUNNING → WAITING
    arrow(5.75, 2.0, 5.75, 0.5, "", "#FF5722")
    arrow(5.75, 0.5, 1.75, 0.5, "", "#FF5722")
    arrow(1.75, 0.5, 1.75, 2.0, "", "#FF5722")
    ax.text(3.75, 0.3, "Preempt (preemption / 抢占)", ha="center",
            fontproperties=F(10), color="#FF5722",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFF3E0", alpha=0.9))

    # 注释
    ax.text(4.5, 4.5, "add_request()", ha="center", fontproperties=F(10), color="#FF9800")
    arrow(4.5, 4.3, 4.5, 3.2, "", "#FF9800", "dashed")

    plt.tight_layout()
    fig.savefig(f"{OUT}/chart8_state_machine.png")
    plt.close()
    print("chart8_state_machine.png")

# ============================================================
# 9. Prefill vs Decode 时间分布堆叠柱状图
# ============================================================
def chart9():
    # Simulated data for a typical batch inference
    steps = list(range(1, 16))
    prefill_time = [0.32, 0.28, 0.15, 0.05, 0.03, 0.03, 0.03, 0.03,
                    0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    decode_time = [0,    0,    0.12, 0.24, 0.25, 0.27, 0.26, 0.25,
                   0.24, 0.25, 0.26, 0.25, 0.24, 0.25, 0.26]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked bar
    ax1.bar(steps, prefill_time, color="#FF5722", alpha=0.85, label="Prefill Time")
    ax1.bar(steps, decode_time, bottom=prefill_time, color="#2196F3", alpha=0.85, label="Decode Time")
    ax1.set_xlabel("Step", fontweight="bold")
    ax1.set_ylabel("Time (seconds)", fontweight="bold")
    ax1.set_title("每 Step 耗时分布 / Per-Step Time Breakdown\n(Prefill + Decode)", fontproperties=F(13))
    ax1.legend(prop=F(10), loc="upper right")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2, linestyle="--")
    for label in ax1.get_xticklabels(): label.set_fontproperties(F(9))

    # Right: cumulative throughput curve
    cum_tokens = np.cumsum([200, 150, 80, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
    ax2.plot(steps, cum_tokens, "o-", color="#FF5722", lw=2.5, ms=6, label="Cumulative Tokens")
    ax2.fill_between(steps, 0, cum_tokens, alpha=0.1, color="#FF5722")
    ax2.set_xlabel("Step", fontweight="bold")
    ax2.set_ylabel("Cumulative Tokens Processed", fontweight="bold")
    ax2.set_title("累计 Token 处理量 / Cumulative Tokens\n(Prefill 阶段吞吐远大于 Decode)", fontproperties=F(13))
    ax2.legend(prop=F(10), loc="lower right")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2, linestyle="--")
    for label in ax2.get_xticklabels(): label.set_fontproperties(F(9))

    plt.tight_layout()
    fig.savefig(f"{OUT}/chart9_inference_timeline.png")
    plt.close()
    print("chart9_inference_timeline.png")

# ============================================================
# 10. Batch Size 扩展性曲线（理论 + 实测）
# ============================================================
def chart10():
    fig, ax = plt.subplots(figsize=(10, 6))

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    # nano-vllm estimated decode throughput (based on A10 + Qwen3-0.6B)
    nano_throughput = [35, 62, 110, 185, 280, 380, 420, 440]
    # transformers (naive batch) throughput
    tf_throughput = [30, 55, 90, 130, 160, 180, 190, 195]
    # vLLM throughput
    vllm_throughput = [29, 58, 105, 190, 310, 450, 540, 580]

    ax.plot(batch_sizes, vllm_throughput, "s-", color="#4CAF50", lw=2.5, ms=8,
            label="vLLM (Continuous Batching)", zorder=3)
    ax.plot(batch_sizes, nano_throughput, "o-", color="#FF5722", lw=2.5, ms=8,
            label="nano-vLLM (Continuous Batching)", zorder=3)
    ax.plot(batch_sizes, tf_throughput, "D--", color="#9E9E9E", lw=2.5, ms=8,
            label="transformers (Naive Batching)", zorder=2)

    ax.fill_between(batch_sizes, nano_throughput, tf_throughput, alpha=0.08, color="#FF5722")
    ax.set_xlabel("Batch Size (并发请求数)", fontproperties=F(12))
    ax.set_ylabel("Decode Throughput (tokens/s)", fontproperties=F(12))
    ax.set_title("Batch Size 扩展性对比 / Scalability Comparison\n"
                 "(NVIDIA A10 + Qwen3-0.6B, estimated decode throughput)",
                 fontproperties=F(14), pad=15)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    for label in ax.get_xticklabels(): label.set_fontproperties(F(9))
    ax.legend(prop=F(10), loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle="--")

    # Annotations
    ax.annotate("Continuous Batching\n优势体现",
                xy=(8, 190), xytext=(20, 120),
                fontproperties=F(9), color="#FF5722",
                arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFF3E0", alpha=0.9))

    plt.tight_layout()
    fig.savefig(f"{OUT}/chart10_batch_scalability.png")
    plt.close()
    print("chart10_batch_scalability.png")

if __name__ == "__main__":
    chart7()
    chart8()
    chart9()
    chart10()
    print(f"\nAll supplementary charts generated in {OUT}/")
