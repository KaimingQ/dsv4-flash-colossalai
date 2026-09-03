# 单测: EpDeepseekV4MoE._grouped_local_forward (grouped GEMM + 8 倍数 pad) 与逐专家循环路径的数值一致性。
# 不需要分布式/大显存, 单卡即可验证 pad/unpad/offs/空组等边界逻辑。
# 用法: CUDA_VISIBLE_DEVICES=0 python scripts/test_grouped_forward.py
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, "third_party/ColossalAI")
from colossalai.shardformer.modeling.deepseek_v4 import EpDeepseekV4MoE

torch.manual_seed(0)
dev, bf16 = "cuda", torch.bfloat16
E_LOCAL, H, I = 32, 4096, 2048
LIMIT = 10.0


def make_block(total, counts_list):
    """构造最小化的 EpDeepseekV4MoE (绕过 ParallelModule.__init__)"""
    block = EpDeepseekV4MoE.__new__(EpDeepseekV4MoE)
    block.experts_per_rank = E_LOCAL
    block.moe_dp_size = 1
    block.ep_size = 8
    block.limit = LIMIT
    # 逐专家权重 (loop 参考路径)
    w_gu = (torch.randn(E_LOCAL, 2 * I, H, device=dev, dtype=bf16) * 0.02)
    w_dn = (torch.randn(E_LOCAL, H, I, device=dev, dtype=bf16) * 0.02)
    experts = SimpleNamespace(limit=LIMIT, act_fn=F.silu)
    experts.experts = [SimpleNamespace(gate_up_proj=w_gu[e], down_proj=w_dn[e]) for e in range(E_LOCAL)]
    block.experts = experts
    # fused 布局 (grouped 路径): 与逐专家权重相同
    block.fused_gate_up = w_gu.clone()
    block.fused_down = w_dn.clone()
    block.gathered = torch.randn(total, H, device=dev, dtype=bf16)
    block.counts = counts_list
    return block


def loop_forward(block):
    """ep_experts_forward 的专家循环段 (grouped 路径不存在时的原实现)"""
    gathered, cl = block.gathered, block.counts
    outs = []
    s = 0
    for e in range(E_LOCAL):
        n = cl[e]
        if n == 0:
            continue
        t = gathered[s : s + n]
        gu = F.linear(t, block.experts.experts[e].gate_up_proj)
        g, u = gu.chunk(2, dim=-1)
        outs.append(F.linear(F.silu(g.clamp(max=LIMIT)) * u.clamp(min=-LIMIT, max=LIMIT), block.experts.experts[e].down_proj))
        s += n
    return torch.cat(outs) if outs else gathered[:0]


cases = [
    ("mixed arbitrary counts", [17, 3, 0, 41, 8, 25, 63, 12, 9, 7, 40, 31, 1, 55, 22, 36, 44, 5, 14, 28, 50, 2, 19, 47, 10, 60, 33, 6, 21, 38, 15, 49]),
    ("all empty except one", [0] * 8 + [96] + [0] * 23),
    ("all multiples of 8", [8] * 16 + [0] * 15 + [24]),
    ("all zero", [0] * E_LOCAL),
]
ok = True
for name, cl in cases:
    total = sum(cl)
    block = make_block(total, cl)
    ref = loop_forward(block)
    got = block._grouped_local_forward(block.gathered, cl)
    d = (ref.float() - got.float()).abs()
    shape_ok = got.shape == ref.shape
    num_ok = d.max().item() < 0.08 if d.numel() else True
    status = "OK" if shape_ok and num_ok else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"[{name}] total={total} shape={tuple(got.shape)} max={d.max().item() if d.numel() else 0:.5f} "
          f"mean={d.mean().item() if d.numel() else 0:.7f} (ref_absmean={ref.float().abs().mean().item() if ref.numel() else 0:.4f}) -> {status}")

print("ALL PASS" if ok else "HAS FAILURES")
