#!/usr/bin/env python3
"""EP all-to-all 内部状态核查: 复刻 ep_experts_forward 的 dispatch/combine,
逐行对比 combine 回来的结果与本地直接计算的专家输出, 定位顺序错位模式。
用法(容器内): torchrun --nproc_per_node=8 scripts/debug_ep_a2a.py --layer 5
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

from colossalai.moe._operation import all_to_all_uneven
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
    eper = ne // world

    blk = DeepseekV4SparseMoeBlock(cfg, L)
    fake_layer = types.SimpleNamespace(mlp=blk)
    fake_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[fake_layer]))
    convert_fused_experts_to_modulelist(fake_model)
    missing, unexpected = blk.load_state_dict(sd, assign=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    blk = blk.to(dev, dtype=torch.bfloat16)
    ep_group = dist.group.WORLD
    dp_group = dist.new_group([rank])
    ep_blk = EpDeepseekV4MoE.from_native_module(blk, moe_dp_group=dp_group, ep_group=ep_group)

    g = torch.Generator(device="cpu").manual_seed(1234)
    B, S = 2, 64
    x = torch.randn(B, S, cfg.hidden_size, generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    ids = torch.randint(0, cfg.vocab_size, (B, S), generator=g)

    with torch.no_grad():
        flat = x.view(-1, cfg.hidden_size)
        if blk.is_hash:
            _, weights, indices = blk.gate(x, ids)
        else:
            _, weights, indices = blk.gate(x)
        k = indices.shape[-1]

        # ---- 参考: 每个 (token, expert) 对的正确输出 (全部在本地算) ----
        flat_idx = indices.reshape(-1)
        sort_order = flat_idx.argsort(stable=True)
        ref_pairs = {}  # 全局 pair 位置 -> 专家输出
        local_start, local_end = rank * eper, (rank + 1) * eper
        for p in range(flat_idx.numel()):
            e = int(flat_idx[p])
            if local_start <= e < local_end:
                tok = flat[p // k].unsqueeze(0)
                ref_pairs[p] = ep_blk.experts.experts[e](tok).squeeze(0)

        # ---- 复刻 ep_experts_forward 的 dispatch ----
        counts = torch.bincount(flat_idx, minlength=ne)
        counts_grouped = counts.view(world, eper).sum(dim=1)
        recv_counts = torch.empty_like(counts_grouped)
        dist.all_to_all_single(recv_counts, counts_grouped, group=ep_group)
        send_splits = counts_grouped.tolist()
        recv_splits = recv_counts.tolist()

        sorted_tokens = flat[sort_order // k]
        sorted_expert = flat_idx[sort_order]

        gathered, _ = all_to_all_uneven(sorted_tokens, send_splits, recv_splits, ep_group)
        expert_gathered, _ = all_to_all_uneven(sorted_expert.unsqueeze(-1), send_splits, recv_splits, ep_group)
        expert_gathered = expert_gathered.squeeze(-1) - local_start

        # 输入侧: 收到的 token 与源位置是否一致 (用唯一指纹验证)
        fingerprint = torch.arange(flat.shape[0], device=dev, dtype=torch.float64).unsqueeze(-1) * 7919.0
        fp_sorted = fingerprint[sort_order // k]
        fp_gathered, _ = all_to_all_uneven(fp_sorted, send_splits, recv_splits, ep_group)

        # 本地计算收到的每个 (token, local_expert) 输出
        local_out_rows = []
        for i in range(gathered.shape[0]):
            e_local = int(expert_gathered[i])
            local_out_rows.append(ep_blk.experts.experts[e_local + local_start](gathered[i].unsqueeze(0)).squeeze(0))
        local_out = torch.stack(local_out_rows, dim=0)

        # combine 回传
        combined, _ = all_to_all_uneven(local_out, recv_splits, send_splits, ep_group)
        fp_back, _ = all_to_all_uneven(fp_gathered, recv_splits, send_splits, ep_group)

        # combined[i] 应对应 sort_order[i] 这个 pair; 用指纹找到其原始全局 pair 位置
        fp_flat = fp_back.squeeze(-1)          # 排序后空间里的指纹
        # 指纹值 = 原 token_idx*7919; 但同一 token 的多个 pair 指纹相同, 还需按 expert 区分
        # 直接按 pair 位置检查: combined[i] 与 ref_pairs[sort_order[i]] 对比
        bad = 0
        maxdiff_bad = 0.0
        ok_diffs = []
        for i in range(combined.shape[0]):
            p = int(sort_order[i])
            ref = ref_pairs.get(p)
            if ref is None:
                continue  # 该 pair 的专家不在本 rank (不应发生: 本 rank 发出的 pair 都回到本 rank)
            d = (combined[i].float() - ref.float()).abs().max().item()
            if d > 0.5:
                bad += 1
                maxdiff_bad = max(maxdiff_bad, d)
            else:
                ok_diffs.append(d)
        import statistics
        print(
            f"[rank {rank}] pairs_checked={len(ok_diffs)+bad} bad(>0.5)={bad} "
            f"ok_maxdiff={max(ok_diffs) if ok_diffs else -1:.4f} bad_maxdiff={maxdiff_bad:.4f}",
            flush=True,
        )
        # 指纹核对: 发送-接收是否对齐
        fp_expect = fp_sorted  # combine 后应恢复发送顺序
        fp_ok = (fp_back.squeeze(-1) - fp_expect.squeeze(-1)).abs().max().item()
        print(f"[rank {rank}] fingerprint_back_maxdiff={fp_ok:.4f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
