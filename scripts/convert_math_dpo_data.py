#!/usr/bin/env python3
"""数学领域 DPO 偏好数据构建:
以 NuminaMath-CoT 为源, 同一问题的"高质量多源解答"作为 chosen,
构造规则: 取带完整推导步骤的解答为 chosen, 仅有简短结论或格式残缺的为 rejected。
输出: HF arrow 格式 (load_from_disk), 字段:
    chosen_input_ids / chosen_loss_mask / rejected_input_ids / rejected_loss_mask

用法(容器内):
    python scripts/convert_math_dpo_data.py --max_samples 2000 --output data/dpo_public
"""
import argparse
import os

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache")

import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
MODEL = os.path.join(MODEL_ROOT, "DeepSeek-V4-Flash-BF16-v2")
MAX_CHARS = 2500


def build_pairs(max_samples):
    """同一问题取两条不同来源解答构成偏好对"""
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    from collections import defaultdict

    by_problem = defaultdict(list)
    for row in ds:
        p, s, src = row.get("problem", ""), row.get("solution", ""), row.get("source", "")
        if not p or not s or len(s) > MAX_CHARS:
            continue
        by_problem[p].append((s, src))
        if sum(len(v) >= 2 for v in by_problem.values()) >= max_samples:
            break
    pairs = []
    for p, sols in by_problem.items():
        if len(sols) < 2:
            continue
        # 偏好规则: 更长且含推导结构 (boxed/therefore/step) 者为 chosen
        def quality(s):
            score = len(s) / MAX_CHARS
            for kw in ["\\boxed", "therefore", "Step", "步骤", "所以", "because"]:
                if kw.lower() in s.lower():
                    score += 0.5
            return score

        sols.sort(key=lambda t: quality(t[0]), reverse=True)
        if quality(sols[0][0]) <= quality(sols[1][0]):
            continue
        pairs.append((p, sols[0][0], sols[1][0]))
        if len(pairs) >= max_samples:
            break
    return pairs


def tokenize_pair(tok, problem, chosen, rejected, max_len):
    def render(answer):
        msgs = [{"role": "user", "content": problem}, {"role": "assistant", "content": answer}]
        return tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"]

    def prompt_len():
        msgs = [{"role": "user", "content": problem}]
        return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"])

    plen = prompt_len()
    out = {}
    for tag, ans in [("chosen", chosen), ("rejected", rejected)]:
        ids = render(ans)[:max_len]
        mask = [False] * min(plen, len(ids)) + [True] * max(0, len(ids) - plen)
        out[f"{tag}_input_ids"] = ids
        out[f"{tag}_loss_mask"] = mask
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--output", type=str, default="data/dpo_public")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    pairs = build_pairs(args.max_samples)
    print(f"偏好对: {len(pairs)}")
    records = [tokenize_pair(tok, p, c, r, args.max_len) for p, c, r in pairs]
    ds = Dataset.from_list(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"保存 -> {args.output} ({len(ds)} 条)")


if __name__ == "__main__":
    main()
