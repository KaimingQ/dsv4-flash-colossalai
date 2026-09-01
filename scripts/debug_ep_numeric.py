#!/usr/bin/env python3
"""层级数值校验: 同一真实权重层, 原生 DeepseekV4SparseMoeBlock vs EpDeepseekV4MoE(EP=8)
用法(容器内): torchrun --nproc_per_node=8 scripts/debug_ep_numeric.py --layer 5 [--layer 0]
"""
import argparse
import json
import os
import types

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoConfig

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"

MODEL = os.environ.get("MODEL", "/home/shared/deepseek-ai/DeepSeek-V4-Flash-BF16-v2")
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)

from colossalai.shardformer.modeling.deepseek_v4 import (
    EpDeepseekV4MoE,
    convert_fused_experts_to_modulelist,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def load_layer_mlp_sd(layer_i: int):
    prefix = f"model.layers.{layer_i}.mlp."
    keys = [k for k in idx if k.startswith(prefix)]
    files = {idx[k] for k in keys}
    sd = {}
    for fn in files:
        with safe_open(os.path.join(MODEL, fn), framework="pt", device="cpu") as f:
            for k in keys:
                if k in f.keys():
                    sd[k[len(prefix):]] = f.get_tensor(k)
    return sd


def build_native_fused(layer_i: int, sd):
    blk = DeepseekV4SparseMoeBlock(cfg, layer_i)
    # 逐专家键 -> 融合 3D 张量
    ne = cfg.n_routed_experts
    gu = torch.stack([sd[f"experts.experts.{e}.gate_up_proj"] for e in range(ne)], dim=0)
    dp = torch.stack([sd[f"experts.experts.{e}.down_proj"] for e in range(ne)], dim=0)
    sd2 = dict(sd)
    for k in list(sd2):
        if k.startswith("experts.experts."):
            del sd2[k]
    sd2["experts.gate_up_proj"] = gu
    sd2["experts.down_proj"] = dp
    missing, unexpected = blk.load_state_dict(sd2, assign=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return blk.to(dev, dtype=torch.bfloat16)


def build_ep(layer_i: int, sd):
    blk = DeepseekV4SparseMoeBlock(cfg, layer_i)
    fake_layer = types.SimpleNamespace(mlp=blk)
    fake_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[fake_layer]))
    convert_fused_experts_to_modulelist(fake_model)  # 融合->逐专家(内部拷贝权重)
    missing, unexpected = blk.load_state_dict(sd, assign=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    blk = blk.to(dev, dtype=torch.bfloat16)
    ep_group = dist.group.WORLD
    dp_group = dist.new_group([rank])
    ep_blk = EpDeepseekV4MoE.from_native_module(blk, moe_dp_group=dp_group, ep_group=ep_group)
    return ep_blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    args = ap.parse_args()
    L = args.layer
    sd = load_layer_mlp_sd(L)
    native = build_native_fused(L, sd)
    ep = build_ep(L, sd)

    g = torch.Generator(device="cpu").manual_seed(1234)
    B, S = 2, 64
    x = torch.randn(B, S, cfg.hidden_size, generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    ids = torch.randint(0, cfg.vocab_size, (B, S), generator=g)

    with torch.no_grad():
        y_ref = native(x, input_ids=ids)
        y_ep = ep(x, input_ids=ids)
    diff = (y_ref.float() - y_ep.float()).abs()
    rel = diff / (y_ref.float().abs() + 1e-6)
    if rank == 0:
        print(
            f"[layer {L}] max_abs_diff={diff.max().item():.6f} mean_abs_diff={diff.mean().item():.6f} "
            f"ref_absmean={y_ref.float().abs().mean().item():.6f} max_rel={rel.max().item():.4f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
