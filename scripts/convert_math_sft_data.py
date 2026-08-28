#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一领域数据准备: 数学推理
SFT 数据源: AI-MO/NuminaMath-CoT (数学题目 + 思维链解答), 备选 tatsu-lab/alpaca
输出: lora_finetune.py 要求的 jsonl 会话格式 (每行一个消息数组)

用法(容器内):
    python scripts/convert_math_sft_data.py --max_samples 5000 --output data/sft_public.jsonl
"""
import argparse
import json
import os
import re

# 默认直连 HF (容器已配代理); 若代理不可用可设 HF_ENDPOINT 镜像后重试
# HF 缓存放在本地盘 (NFS 上 mmap 会 core dump)
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache")

from datasets import load_dataset

# 过滤过长样本 (训练时按 512~4096 长度截断, 过长样本浪费)
MAX_CHARS = 3000


def convert_numina(max_samples: int, skip: int = 0):
    """AI-MO/NuminaMath-CoT: problem / solution (含 CoT); skip 用于分批续拉"""
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    if skip > 0:
        ds = ds.skip(skip)
    n = 0
    for row in ds:
        if n >= max_samples:
            break
        problem, solution = row.get("problem", ""), row.get("solution", "")
        if not problem or not solution or len(problem) + len(solution) > MAX_CHARS:
            continue
        yield [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
        n += 1


def convert_alpaca_math(max_samples: int):
    """备选: alpaca 中数学相关子集 (粗过滤)"""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    kw = re.compile(r"math|calculat|number|equation|algebra|geometry|sum of|solve", re.I)
    n = 0
    for row in ds:
        if n >= max_samples:
            break
        text = (row["instruction"] or "") + " " + (row["input"] or "")
        if not kw.search(text):
            continue
        user = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
        if len(user) + len(row["output"]) > MAX_CHARS:
            continue
        yield [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}]
        n += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["numina", "alpaca_math"], default="numina")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=500, help="分批调用, 规避流式长连接崩溃")
    parser.add_argument("--single_batch", action="store_true", help="子进程模式: 单批直接写出")
    parser.add_argument("--skip", type=int, default=0, help="流式偏移 (分批续拉)")
    parser.add_argument("--output", type=str, default="data/sft_public.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.source == "numina" and not args.single_batch:
        # 分批子进程调用自身, 每批独立进程, 规避流式迭代中途 core dump
        import subprocess
        import sys

        done = 0
        part_i = 0
        with open(args.output, "w", encoding="utf-8") as f:
            while done < args.max_samples:
                take = min(args.batch, args.max_samples - done)
                part = f"{args.output}.part{part_i}"
                r = subprocess.run(
                    [sys.executable, __file__, "--source", "numina", "--single_batch",
                     "--max_samples", str(take), "--skip", str(done), "--output", part],
                    check=False,
                    env=os.environ,
                )
                # 注: 流式迭代线程在进程退出时可能触发 abort (rc=-6), 但文件已完整写出, 按文件内容判定成败
                if os.path.exists(part):
                    with open(part, encoding="utf-8") as pf:
                        lines = pf.readlines()
                    f.writelines(lines)
                    os.remove(part)
                    done += len(lines)
                    print(f"[batch] 已收集 {done}/{args.max_samples} (rc={r.returncode})", flush=True)
                    if len(lines) < take:
                        break
                else:
                    print(f"[warn] 批次无产出 (rc={r.returncode}), 重试一次")
                    if r.returncode not in (-6, 134):
                        break
                part_i += 1
        print(f"完成: 共 {done} 条 -> {args.output}")
        return

    try:
        gen = convert_numina(args.max_samples, skip=args.skip) if args.source == "numina" else convert_alpaca_math(args.max_samples)
        n = 0
        with open(args.output, "w", encoding="utf-8") as f:
            for msgs in gen:
                f.write(json.dumps(msgs, ensure_ascii=False) + "\n")
                n += 1
        print(f"写入 {n} 条样本 ({args.source}) -> {args.output}")
    except Exception as e:
        raise SystemExit(f"数据准备失败: {e}")


if __name__ == "__main__":
    main()
