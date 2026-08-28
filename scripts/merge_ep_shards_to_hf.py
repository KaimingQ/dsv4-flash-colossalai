#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 ColossalAI moe 插件保存的 EP 分片 (.bin) 合并导出为 HuggingFace 融合布局:

- 输入: output/dsv4-dpo/modeling/pytorch_model-stage-*.bin (非重叠键分片, 逐专家键)
- 转换: model.layers.X.mlp.experts.experts.N.gate_up_proj (256 个 [2I, H])
        -> model.layers.X.mlp.experts.gate_up_proj ([E, 2I, H]) (down_proj 同理)
- 校验: 导出键集合与 transformers meta 模型的参数/缓冲区集合严格一致
- 输出: safetensors 分片 + index + config/tokenizer, 可被 from_pretrained 直接加载

用法(容器内):
    python scripts/merge_ep_shards_to_hf.py \
        --input output/dsv4-dpo/modeling --output output/dsv4-dpo-hf
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
CONFIG_SRC = os.path.join(MODEL_ROOT, "DeepSeek-V4-Flash-BF16-v2")

EXPERT_RE = re.compile(r"^(?P<prefix>.*)\.experts\.experts\.(?P<eid>\d+)\.(?P<name>gate_up_proj|down_proj)$")


def expected_keys(config_src):
    cfg = AutoConfig.from_pretrained(config_src, trust_remote_code=True)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    keys = {n for n, _ in m.named_parameters()}
    # 持久化缓冲区 (如 sinks) 也在 checkpoint 中; 排除非持久化的 (rotary inv_freq)
    for name, mod in m.named_modules():
        npb = getattr(mod, "_non_persistent_buffers_set", set())
        for bn, _ in mod.named_buffers(recurse=False):
            if bn not in npb:
                keys.add(f"{name}.{bn}" if name else bn)
    del m
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard_gb", type=float, default=4.5)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "pytorch_model-*.bin")))
    if not files:  # 兼容 safetensors 分片 (convert_native_to_bf16_v2 产物)
        files = sorted(glob.glob(os.path.join(args.input, "model-*.safetensors")))
    print(f"[merge] {len(files)} 个分片")
    assert files, f"无分片: {args.input}"

    # 1) 汇总全部键 (逐专家键先按组收集)
    expert_groups = defaultdict(dict)  # (prefix, name) -> {eid: tensor}
    plain = {}
    for fi, f in enumerate(files):
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(f, device="cpu")
        else:
            sd = torch.load(f, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            # 运行时缓冲区 (rotary inv_freq 等): 模型初始化自动生成, 不入产物
            if "inv_freq" in k:
                continue
            m = EXPERT_RE.match(k)
            if m:
                expert_groups[(m["prefix"], m["name"])][int(m["eid"])] = v
            else:
                plain[k] = v
        if (fi + 1) % 100 == 0:
            print(f"[merge] 读取 {fi+1}/{len(files)}", flush=True)

    # 2) 逐专家 -> 融合 3D
    for (prefix, name), group in expert_groups.items():
        eids = sorted(group)
        assert eids == list(range(len(eids))), f"专家编号不连续: {prefix}.{name}"
        fused = torch.stack([group[e] for e in eids], dim=0).contiguous()
        plain[f"{prefix}.experts.{name}"] = fused
        for e in eids:
            del group[e]
        print(f"[merge] 融合 {prefix}.experts.{name} -> {list(fused.shape)}", flush=True)

    # 3) 与 HF 预期键校验 (配置优先读输入目录, 兼容截断层数的测试产物)
    cfg_src = args.input if os.path.exists(os.path.join(args.input, "config.json")) else CONFIG_SRC
    expect = expected_keys(cfg_src)
    got = set(plain)
    missing = sorted(expect - got)[:8]
    unexpected = sorted(got - expect)[:8]
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    print(f"[merge] 键校验通过 ({len(got)} 键)")

    # 4) 分片写出
    os.makedirs(args.output, exist_ok=True)
    max_bytes = int(args.shard_gb * 1024**3)
    keys_sorted = sorted(plain)
    bounds, cur, start = [], 0, 0
    for i, k in enumerate(keys_sorted):
        nb = plain[k].numel() * plain[k].element_size()
        if cur + nb > max_bytes and i > start:
            bounds.append(start)
            start, cur = i, 0
        cur += nb
    bounds.append(start)
    total = len(bounds)
    weight_map = {}
    for si, s0 in enumerate(bounds):
        s1 = bounds[si + 1] if si + 1 < total else len(keys_sorted)
        shard = {k: plain[k] for k in keys_sorted[s0:s1]}
        fn = f"model-{si+1:05d}-of-{total:05d}.safetensors"
        print(f"[merge] 保存 {fn} ({len(shard)} 键)", flush=True)
        save_file(shard, os.path.join(args.output, fn), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fn
        for k in shard:
            del plain[k]
    idx = {
        "metadata": {"total_size": None},
        "weight_map": weight_map,
    }
    json.dump(idx, open(os.path.join(args.output, "model.safetensors.index.json"), "w"), indent=1)

    # 5) config / tokenizer (优先用输入目录配置)
    cfg = AutoConfig.from_pretrained(cfg_src, trust_remote_code=True)
    cfg.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(cfg_src, trust_remote_code=True).save_pretrained(args.output)
    print(f"[merge] 完成 -> {args.output}")


if __name__ == "__main__":
    main()
