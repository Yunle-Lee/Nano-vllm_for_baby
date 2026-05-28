#!/usr/bin/env python3
"""nano-vllm for baby — 可视化图表集 (中文字体适配版)"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
import os

# ---- 中文字体 ----
FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
F = lambda s=11: FontProperties(fname=FONT, size=s)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
OUT = "/mnt/workspace/nano-vllm for baby/charts"
os.makedirs(OUT, exist_ok=True)

def make_patch(color, label):
    """创建图例句柄"""
    return Rectangle((0, 0), 1, 1, fc=color, label=label)

# ============================================================
# 1. 性能对比柱状图
# ============================================================
def chart1():
    fig, ax = plt.subplots(figsize=(8, 5))
    frameworks = ["nano-vllm", "transformers\n(HuggingFace)", "vLLM 0.19.1"]
    throughput = [35.0, 30.3, 29.5]
    colors = ["#FF5722", "#2196F3", "#4CAF50"]

    bars = ax.bar(frameworks, throughput, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f} tok/s", ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_ylabel("Decode Throughput (tokens/s)", fontproperties=F(12))
    ax.set_title("Qwen3-0.6B 单请求 Decode 吞吐量对比\n(NVIDIA A10, max_tokens=64)",
                 fontproperties=F(14), pad=15)
    ax.set_ylim(0, 42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    b = bars[0]
    ax.annotate("+15.5% vs vLLM", xy=(b.get_x() + b.get_width() / 2, 37),
                xytext=(b.get_x() + b.get_width() / 2 + 0.7, 39.5),
                fontsize=10, fontweight="bold", color="#FF5722",
                arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1.5), ha="center")
    for label in ax.get_xticklabels():
        label.set_fontproperties(F(11))
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart1_performance.png")
    plt.close()
    print("chart1_performance.png")

# ============================================================
# 2. 代码量分布图
# ============================================================
def chart2():
    modules = {
        "engine/model_runner": 257, "models/qwen3": 182, "layers/linear": 156,
        "engine/block_manager": 120, "engine/scheduler": 92, "engine/sequence": 79,
        "engine/llm_engine": 70, "layers/embed_head": 67, "layers/attention": 58,
        "layers/rotary_embedding": 50, "layers/layernorm": 45, "layers/sampler": 30,
        "utils/loader": 28, "config": 20, "layers/activation": 15, "utils/context": 12,
        "sampling_params": 12, "llm": 5,
    }
    names, lines = list(modules.keys()), list(modules.values())
    cat_colors = []
    for n in names:
        if n.startswith("engine/"): cat_colors.append("#FF5722")
        elif n.startswith("layers/"): cat_colors.append("#2196F3")
        elif n.startswith("models/"): cat_colors.append("#9C27B0")
        elif n.startswith("utils/"): cat_colors.append("#FF9800")
        else: cat_colors.append("#4CAF50")

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, lines, color=cat_colors, edgecolor="white", height=0.65)
    for bar, val in zip(bars, lines):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{val} lines", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Lines of Code", fontweight="bold")
    ax.set_title("nano-vllm 代码量分布（共 ~1300 行）", fontproperties=F(14), pad=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 310)

    ax.legend(handles=[
        make_patch("#FF5722", "engine/ 调度&执行"),
        make_patch("#2196F3", "layers/ 算子层"),
        make_patch("#9C27B0", "models/ 模型定义"),
        make_patch("#FF9800", "utils/ 工具"),
        make_patch("#4CAF50", "顶层配置"),
    ], loc="lower right", prop=F(9), framealpha=0.9)
    for label in ax.get_xticklabels(): label.set_fontproperties(F(9))
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart2_code_distribution.png")
    plt.close()
    print("chart2_code_distribution.png")

# ============================================================
# 3. 推理流水线架构图
# ============================================================
def chart3():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("nano-vllm 推理流水线架构", fontproperties=F(15), pad=20)

    def box(x, y, w, h, text, color, sub=""):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           fc=color, ec="white", lw=1.5, alpha=0.9)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), text,
                ha="center", va="center", color="white", fontproperties=F(10))
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.33, sub,
                    ha="center", va="center", color="white", fontsize=8, alpha=0.85)

    def arrow(x1, y1, x2, y2, c="gray", ds=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2 if not ds else 1.5,
                                    linestyle="dashed" if ds else "solid"))

    # 左列
    box(0.5, 8.3, 2.2, 1.0, "进入", "#333333")
    box(0.5, 6.6, 2.2, 1.0, "Tokenizer", "#607D8B", "encode(prompt) -> IDs")
    box(0.5, 4.9, 2.2, 1.0, "Scheduler", "#FF5722", "schedule(): prefill / decode")
    box(0.5, 3.2, 2.2, 1.0, "ModelRunner", "#2196F3", "prepare -> forward -> sample")
    box(0.5, 1.5, 2.2, 1.0, "出去", "#333333")
    arrow(1.6, 8.3, 1.6, 7.6); arrow(1.6, 6.6, 1.6, 5.9)
    arrow(1.6, 4.9, 1.6, 4.2); arrow(1.6, 3.2, 1.6, 2.5)

    # 中间列
    box(3.8, 5.8, 2.8, 0.8, "BlockManager", "#E91E63", "分配/回收 KV 块 + Prefix Caching")
    box(3.8, 4.7, 2.8, 0.8, "Waiting 队列", "#FFC107", "尚未 prefill 的序列")
    box(3.8, 3.6, 2.8, 0.8, "Running 队列", "#4CAF50", "正在 decode 的序列")
    arrow(2.7, 5.3, 3.8, 6.2, c="#888", ds=True)
    arrow(2.7, 5.1, 3.8, 5.1, c="#888", ds=True)
    arrow(2.7, 4.4, 3.8, 4.0, c="#888", ds=True)

    # 右侧
    box(7.5, 5.8, 3.0, 0.8, "prepare_prefill()", "#00BCD4",
        "构造 cu_seqlens + slot_mapping")
    box(7.5, 4.5, 3.0, 0.8, "prepare_decode()", "#00BCD4",
        "每条序列 1 token + block_table")
    box(7.5, 2.7, 3.0, 0.8, "CUDA Graph Replay", "#9C27B0",
        "batch 1~512 预录制图")
    box(7.5, 1.2, 3.0, 0.8, "Sampler", "#795548",
        "Gumbel-max trick 采样")

    # PagedAttention 框
    box(11.3, 3.3, 3.5, 1.8, "PagedAttention\n(KV Cache)", "#FF5722",
        "Triton kernel 写 KV\nflash_attn_varlen / with_kvcache")
    arrow(2.7, 3.6, 7.5, 6.2); arrow(2.7, 3.6, 7.5, 4.9)
    arrow(7.5, 4.5, 7.5, 3.5); arrow(7.5, 2.7, 7.5, 2.0)
    arrow(7.5, 5.8, 11.3, 4.2); arrow(7.5, 4.9, 11.3, 4.2)
    arrow(11.3, 3.3, 7.5, 3.1)

    # 循环箭头
    cp = FancyArrowPatch((1.6, 1.8), (0.3, 8.8),
                         connectionstyle="arc3,rad=-0.5",
                         arrowstyle="->", color="#FF9800", lw=2, linestyle="dashed")
    ax.add_patch(cp)
    ax.text(0.05, 5.3, "while not\nfinished", rotation=86,
            fontproperties=F(9), color="#FF9800", ha="center")

    # 底部图例
    ax.legend(handles=[
        make_patch("#FF5722", "调度 & 队列"),
        make_patch("#2196F3", "GPU 执行"),
        make_patch("#E91E63", "KV Cache"),
        make_patch("#9C27B0", "Optimization"),
    ], loc="lower right", prop=F(9))

    plt.tight_layout()
    fig.savefig(f"{OUT}/chart3_architecture.png")
    plt.close()
    print("chart3_architecture.png")

# ============================================================
# 4. 核心特性雷达图
# ============================================================
def chart4():
    categories = ["PagedAttention", "Prefix\nCaching", "Continuous\nBatching",
                  "Tensor\nParallel", "CUDA\nGraph", "Chunked\nPrefill", "Torch\nCompile"]
    N = len(categories)
    nanovllm = np.array([5, 5, 4, 4, 4, 3, 4] + [5], dtype=float)
    vllm = np.array([5, 5, 5, 5, 5, 5, 3] + [5], dtype=float)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.fill(angles, vllm, alpha=0.12, color="#4CAF50")
    ax.plot(angles, vllm, "o-", color="#4CAF50", lw=2, ms=6, label="vLLM")
    ax.fill(angles, nanovllm, alpha=0.22, color="#FF5722")
    ax.plot(angles, nanovllm, "o-", color="#FF5722", lw=2, ms=6, label="nano-vLLM")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5.5); ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("nano-vllm vs vLLM 核心特性覆盖度 (1-5)",
                 fontproperties=F(14), pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), prop=F(11))
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart4_features_radar.png")
    plt.close()
    print("chart4_features_radar.png")

# ============================================================
# 5. PagedAttention 内存布局图
# ============================================================
def chart5():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("PagedAttention — 物理块映射 & Prefix Caching 共享",
                 fontproperties=F(14), pad=15)

    def blk(x, y, color, txt="", alpha=0.88):
        r = FancyBboxPatch((x, y), 1.2, 0.6, boxstyle="round,pad=0.05",
                           fc=color, ec="white", lw=1, alpha=alpha)
        ax.add_patch(r)
        if txt:
            ax.text(x + 0.6, y + 0.3, txt, ha="center", va="center",
                    fontproperties=F(10), color="white" if color != "#FFC107" else "#333")

    # 物理 KV Cache
    ax.text(0.5, 7.5, "GPU 物理 KV Cache（显存）", fontproperties=F(12), color="#333")
    cols = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
            "#9C27B0", "#FF5722", "#00BCD4", "#795548",
            "#607D8B", "#3F51B5"]
    for i in range(10):
        blk(0.3 + i * 1.3, 6.5, cols[i], f"Block {i}")

    # 序列 A
    ax.text(0.5, 5.7, "序列 A（请求 1）block_table:", fontproperties=F(11))
    for i, j in enumerate([0, 3, 1, 7]):
        blk(0.3 + i * 2.3, 5.0, cols[j], f"Block {j}")

    # 序列 B（共享前缀）
    ax.text(0.5, 4.2, "序列 B（请求 2）block_table — Prefix Caching 命中 Block 0 + Block 3",
            fontproperties=F(11))
    for i, j in enumerate([0, 3, 5, 8]):
        blk(0.3 + i * 2.3, 3.5, cols[j], f"Block {j}")

    # 标注共享
    style = dict(fontproperties=F(10), color="#E91E63", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", alpha=0.9))
    ax.text(2.55, 4.7, "ref_count=2", **style)
    ax.text(4.85, 4.7, "ref_count=2", **style)

    # 图例
    ax.legend(handles=[
        make_patch("#2196F3", "命中前缀缓存"),
        make_patch("#E91E63", "新分配块"),
    ], loc="lower right", prop=F(10))

    plt.tight_layout()
    fig.savefig(f"{OUT}/chart5_memory_layout.png")
    plt.close()
    print("chart5_memory_layout.png")

# ============================================================
# 6. 模块依赖关系图
# ============================================================
def chart6():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("nano-vllm 模块依赖关系", fontproperties=F(14), pad=15)

    nodes = {
        "llm.py":       (7.5, 8.2, "#333"),
        "llm_engine":   (7.5, 7.0, "#FF5722"),
        "scheduler":    (10.2, 7.0, "#FF9800"),
        "block_manager":(13.0, 7.0, "#E91E63"),
        "model_runner": (4.8, 7.0, "#2196F3"),
        "config":       (7.5, 5.6, "#4CAF50"),
        "sampling_params":(10.2, 5.6, "#795548"),
        "sequence":     (13.0, 5.6, "#607D8B"),
        "qwen3.py":     (2.5, 5.6, "#9C27B0"),
        "attention":    (1.8, 3.8, "#00BCD4"),
        "sampler":      (4.8, 3.8, "#795548"),
        "linear":       (7.2, 3.8, "#3F51B5"),
        "embed_head":   (10.0, 3.8, "#00BCD4"),
        "rotary_emb":   (13.0, 3.8, "#FFC107"),
        "layernorm":    (1.8, 2.2, "#8BC34A"),
        "activation":   (4.8, 2.2, "#8BC34A"),
        "context":      (7.5, 2.2, "#FF9800"),
        "loader":       (10.5, 2.2, "#607D8B"),
    }

    for name, (x, y, color) in nodes.items():
        r = FancyBboxPatch((x - 0.7, y - 0.25), 1.4, 0.5, boxstyle="round,pad=0.05",
                           fc=color, ec="white", lw=1, alpha=0.9)
        ax.add_patch(r)
        ax.text(x, y, name, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    edges = [
        ("llm.py","llm_engine"),
        ("llm_engine","scheduler"),("llm_engine","model_runner"),("llm_engine","config"),
        ("llm_engine","sequence"),("llm_engine","sampling_params"),
        ("scheduler","block_manager"),("scheduler","config"),("scheduler","sequence"),
        ("block_manager","sequence"),
        ("model_runner","qwen3.py"),("model_runner","sampler"),("model_runner","config"),
        ("model_runner","sequence"),("model_runner","context"),("model_runner","loader"),
        ("qwen3.py","attention"),("qwen3.py","linear"),("qwen3.py","embed_head"),
        ("qwen3.py","rotary_emb"),("qwen3.py","layernorm"),("qwen3.py","activation"),
        ("attention","context"),("embed_head","context"),
    ]
    for s, d in edges:
        sx, sy = nodes[s][0], nodes[s][1] - 0.25
        dx, dy = nodes[d][0], nodes[d][1] + 0.25
        ax.plot([sx, dx], [sy, dy], color="#BDBDBD", lw=0.8, alpha=0.5, zorder=0)

    ax.legend(handles=[
        make_patch("#333", "顶层入口"),
        make_patch("#FF5722", "engine/ 引擎"),
        make_patch("#9C27B0", "models/ 模型"),
        make_patch("#00BCD4", "layers/ 算子"),
        make_patch("#FF9800", "utils/ 工具"),
        make_patch("#4CAF50", "配置 & 参数"),
    ], loc="lower right", prop=F(8))

    plt.tight_layout()
    fig.savefig(f"{OUT}/chart6_dependency.png")
    plt.close()
    print("chart6_dependency.png")


if __name__ == "__main__":
    chart1()
    chart2()
    chart3()
    chart4()
    chart5()
    chart6()
    print(f"\n图表已全部生成到 {OUT}/")
