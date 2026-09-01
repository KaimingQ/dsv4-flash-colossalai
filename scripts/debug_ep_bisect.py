#!/usr/bin/env python3
"""EP MoE 拆解校验:
  A) 单专家前向: V4Expert vs 原生融合切片 (纯计算)
  B) 每 rank 本地分组计算 (绕开 all_to_all) vs 原生逐专家结果
  C) 完整 EP forward vs 原生 (定位 all_to_all 调度)
用法(容器内): torchrun --nproc_per_node=8 scripts/debug_ep_bisect.py --layer 5
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    args = ap.parse_args()
    L = args.layer
    sd = load_layer_mlp_sd(L)
    ne = cfg.n_routed_experts
    eper = ne // dist.get_world_size()

    # ---- 原生参考: 直接用逐专家张量做逐专家计算 (不依赖融合模块) ----
    g = torch.Generator(device="cpu").manual_seed(1234)
    B, S = 2, 64
    x = torch.randn(B, S, cfg.hidden_size, generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    ids = torch.randint(0, cfg.vocab_size, (B, S), generator=g)

    blk = DeepseekV4SparseMoeBlock(cfg, L)
    fake_layer = types.SimpleNamespace(mlp=blk)
    fake_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[fake_layer]))
    convert_fused_experts_to_modulelist(fake_model)
    missing, unexpected = blk.load_state_dict(sd, assign=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    blk = blk.to(dev, dtype=torch.bfloat16)

    with torch.no_grad():
        flat = x.view(-1, cfg.hidden_size)
        if blk.is_hash:
            _, w, ix = blk.gate(x, ids)
        else:
            _, w, ix = blk.gate(x)
        # 原生逐专家参考 (全部专家, 朴素循环)
        ref = blk.experts(flat, ix, w)

    # ---- A) 单专家计算一致性: V4Expert vs 逐专家参考 ----
    # 取本 rank 持有的第一个专家, 用一个确定命中的输入
    e0 = rank * eper
    tok = flat[:8]
    with torch.no_grad():
        out_v4 = blk.experts.experts[e0](tok)
        # 原生融合切片路径
        gu = sd[f"experts.experts.{e0}.gate_up_proj"].to(dev, torch.bfloat16)
        dp = sd[f"experts.experts.{e0}.down_proj"].to(dev, torch.bfloat16)
        gate_up = torch.nn.functional.linear(tok, gu)
        gg, uu = gate_up.chunk(2, dim=-1)
        out_ref = torch.nn.functional.linear(
            torch.nn.functional.silu(gg.clamp(max=10.0)) * uu.clamp(min=-10.0, max=10.0), dp
        )
    if rank == 0:
        dA = (out_v4.float() - out_ref.float()).abs()
        print(f"[A 单专家] max_abs_diff={dA.max().item():.6f} out_absmean={out_ref.float().abs().mean().item():.6f}", flush=True)

    # ---- B) 每 rank 只算本地持有的专家并 all_reduce 汇总, 与原生全量结果对比 ----
    with torch.no_grad():
        local = torch.zeros_like(ref)
        for e in range(rank * eper, (rank + 1) * eper):
            mask = ix == e
            if not mask.any():
                continue
            t_idx, pos = mask.nonzero(as_tuple=True)
            o = blk.experts.experts[e](flat[t_idx]) * w[t_idx, pos, None]
            local.index_add_(0, t_idx, o.to(local.dtype))
        dist.all_reduce(local)
    if rank == 0:
        dB = (local.float() - ref.float()).abs()
        print(f"[B 本地分组+allreduce] max_abs_diff={dB.max().item():.6f} ref_absmean={ref.float().abs().mean().item():.6f}", flush=True)

    # ---- C) 完整 EP forward ----
    ep_group = dist.group.WORLD
    dp_group = dist.new_group([rank])
    ep_blk = EpDeepseekV4MoE.from_native_module(blk, moe_dp_group=dp_group, ep_group=ep_group)
    with torch.no_grad():
        y_ep = ep_blk(x, input_ids=ids)
        y_ref_full = ref.view(B, S, cfg.hidden_size) + blk.shared_experts(x)
    dC = (y_ep.float() - y_ref_full.float()).abs()
    if rank == 0:
        print(f"[C 完整EP] max_abs_diff={dC.max().item():.6f} mean={dC.mean().item():.6f} ref_absmean={y_ref_full.float().abs().mean().item():.6f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
