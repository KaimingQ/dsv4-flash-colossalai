#!/usr/bin/env python3
"""基线 NLL 校验: HF 原生前向 + coati 相同数据模板, 给出 SFT 第 0 步应有的 loss。
用法(容器内): python scripts/debug_base_loss.py [--model .../DeepSeek-V4-Flash-BF16-v2-fused]
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coati.dataset.loader import apply_chat_template_and_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/shared/deepseek-ai/DeepSeek-V4-Flash-BF16-v2-fused")
    ap.add_argument("--data", default="data/sft_smoke.jsonl")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--num", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        experts_implementation="eager",  # grouped_mm 内核在本环境崩溃, 用原生逐专家循环
    )
    model.eval()
    print(f"[eval] 模型就绪: {args.model}", flush=True)

    rows = [json.loads(l) for l in open(args.data)][: args.num]
    losses = []
    with torch.no_grad():
        for r in rows:
            t = apply_chat_template_and_mask(tok, r, args.max_length, "")
            batch = {k: v.unsqueeze(0).to(model.device) for k, v in t.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            losses.append(out.loss.item())
            print(f"  sample loss={out.loss.item():.4f} ntok={(t['labels'] != -100).sum().item()}", flush=True)
    print(f"[eval] 基座模型基线 NLL: mean={sum(losses)/len(losses):.4f} ({len(losses)} 条)", flush=True)


if __name__ == "__main__":
    main()
