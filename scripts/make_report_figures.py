#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成报告 (docs/06_report.md) 所需图表 -> docs/figures/

输入:
  - 训练日志 (tqdm \r 分隔): logs/full_run.log / orpo_run.log
  - 显存: 日志中 "Max device memory usage: X MB"
输出:
  - docs/figures/loss_curves.png    各阶段 loss 曲线
  - docs/figures/memory_usage.png    各阶段单卡峰值显存 (对比 96GB 上限)
用法(容器内): python scripts/make_report_figures.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = "docs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# tqdm 行: "Epoch 0:  5%|...| 16/31 [00:34<..., loss=131.00, lr=...]"
# ORPO tqdm 描述: "Epoch 1/1 Loss: 4.0726:  37%|...| 50/133 [15:21<25:30, 18.43s/it]"
LOSS_RE = re.compile(r"Epoch (\d+)(?:/\d+)?:.*?(\d+)/(\d+) \[[^\]]*loss=([\d.]+)")
# DPO/SimPO 的 tqdm 描述为 "Epoch 1/1: ... train/loss=x" (epoch 从 1 计数),
# 走通用 LOSS_RE 会被折算成第二 epoch 造成 step 偏移, 单独匹配取步内序号
LOSS_RE_DPO = re.compile(r"Epoch \d+/\d+:.*?(\d+)/(\d+) \[[^\]]*train/loss=([\d.]+)")
# ORPO 行含大量 \r 撕裂, 用无 epoch 前缀的宽松匹配 (单 epoch)
LOSS_RE_ORPO = re.compile(r"Loss: ([\d.]+):\s*\d+%\|[^\|]*\|\s*(\d+)/(\d+) \[")
MEM_RE = re.compile(r"Max (?:device|CUDA) memory usage: ([\d.]+) MB|Booster init max CUDA memory: ([\d.]+) MB")


def read_log(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore").replace("\r", "\n")


def parse_losses(path):
    """返回 [(global_step, loss)], 多 epoch 时按每 epoch 步数拼接"""
    pts = []
    for line in read_log(path).split("\n"):
        m = LOSS_RE_DPO.search(line)
        if m:
            cur, total, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
            pts.append((cur, loss))
            continue
        m = LOSS_RE.search(line)
        if m:
            ep, cur, total, loss = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            pts.append((ep * total + cur, loss))
            continue
        m = LOSS_RE_ORPO.search(line)
        if m:
            loss, cur, total = float(m.group(1)), int(m.group(2)), int(m.group(3))
            pts.append((cur, loss))
    # 去重 (同一 step 多次刷新取最后)
    dedup = {}
    for s, l in pts:
        dedup[s] = l
    return sorted(dedup.items())


def parse_mem(path):
    vals = [float(a or b) for a, b in MEM_RE.findall(read_log(path))]
    return max(vals) / 1024 if vals else None  # GB


def smooth(ys, w=10):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - w + 1)
        out.append(sum(ys[lo : i + 1]) / (i - lo + 1))
    return out


# ---- 1) loss 曲线 ----
RUNS = [
    ("SFT (LoRA)", "logs/full_run.log"),
    ("DPO/SimPO (SimPO loss)", "logs/dpo_run.log"),
    ("ORPO", "logs/orpo_run.log"),
]
fig, axes = plt.subplots(1, len(RUNS), figsize=(5.2 * len(RUNS), 3.6), sharey=False)
if len(RUNS) == 1:
    axes = [axes]
for ax, (name, path) in zip(axes, RUNS):
    if not os.path.exists(path):
        ax.set_title(f"{name} (无日志)")
        continue
    pts = parse_losses(path)
    if not pts:
        ax.set_title(f"{name} (未解析到 loss)")
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, alpha=0.35, lw=0.8)
    ax.plot(xs, smooth(ys), lw=1.8, color="tab:red")
    ax.set_title(f"{name} ({len(pts)} pts)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
fig.suptitle("284B DeepSeek-V4-Flash @ 8xH20 (ColossalAI EP=8 + ZeRO1) loss curves")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/loss_curves.png", dpi=150)
print(f"saved {FIG_DIR}/loss_curves.png")

# ---- 2) 显存对比 ----
mems = {}
for name, path in [("SFT", "logs/full_run.log"), ("DPO/SimPO", "logs/dpo_run.log"), ("ORPO", "logs/orpo_run.log")]:
    if os.path.exists(path):
        v = parse_mem(path)
        if v:
            mems[name] = v
if mems:
    fig, ax = plt.subplots(figsize=(1.6 * len(mems) + 2, 3.8))
    bars = ax.bar(list(mems), list(mems.values()), color="tab:green", alpha=0.85)
    ax.axhline(96, color="red", ls="--", lw=1.2, label="H20 96GB limit")
    for b, v in zip(bars, mems.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}GB", ha="center")
    ax.set_ylabel("Peak GPU memory per card (GB)")
    ax.set_title("284B post-training peak memory (8xH20, EP=8 + ZeRO1)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/memory_usage.png", dpi=150)
    print(f"saved {FIG_DIR}/memory_usage.png")
print("done")
