# 探查训练快路径可行性 (单卡, GPU 空闲时运行):
#   A. SDPA EFFICIENT_ATTENTION 后端对 head_dim=512 bf16 + additive mask + GQA 的支持与 LSE 返回
#   B. "SDPA + sinks 重标定" 与 HF eager_attention_forward 的数值等价性
#   C. torch._grouped_mm 对逐专家小 GEMM 循环的替代可行性与加速比
# 用法: python scripts/probe_fastkernels.py
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
bf16 = torch.bfloat16

B, H, S, D = 2, 64, 512, 512  # V4-Flash attention 形状
print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

# ---------------- A. efficient attention D=512 ----------------
q = torch.randn(B, H, S, D, device=dev, dtype=bf16)
k = torch.randn(B, 1, S, D, device=dev, dtype=bf16)  # MQA: 单 KV 头
v = torch.randn(B, 1, S, D, device=dev, dtype=bf16)
mask = torch.zeros(B, S, S, device=dev, dtype=bf16)
causal = torch.tril(torch.ones(S, S, device=dev, dtype=torch.bool))
window = torch.arange(S, device=dev)[None, :] >= torch.arange(S, device=dev)[:, None] - 127
mask = torch.where(causal & window, 0.0, torch.finfo(bf16).min)[None]

from torch.nn.attention import SDPBackend, sdpa_kernel

ok_eff = False
try:
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True, scale=D**-0.5)
    torch.cuda.synchronize()
    ok_eff = True
    print(f"[A] EFFICIENT_ATTENTION D=512 + GQA + additive mask: OK out={tuple(out.shape)}")
except Exception as e:
    print(f"[A] EFFICIENT_ATTENTION failed: {type(e).__name__}: {e}")

# LSE via private aten op (sinks 重标定需要)
lse = None
if ok_eff:
    try:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            r = torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k.expand(B, H, S, D), v.expand(B, H, S, D), mask, False, 0.0, False, scale=D**-0.5
            )
        out2, lse = r[0], r[1]
        print(f"[A] aten efficient LSE: OK lse={tuple(lse.shape)} {lse.dtype}")
    except Exception as e:
        print(f"[A] aten efficient LSE failed: {type(e).__name__}: {e}")

# ---------------- B. sinks 重标定 vs eager ----------------
def eager_ref(query, key, value, attention_mask, scaling, sinks):
    key = key.expand(query.shape[0], query.shape[1], key.shape[2], key.shape[3])
    value = value.expand(query.shape[0], query.shape[1], value.shape[2], value.shape[3])
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    sk = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], 1)
    combined = torch.cat([attn_weights, sk], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
    scores = probs[..., :-1]
    return torch.matmul(scores, value)

def sdpa_sinks(query, key, value, attention_mask, scaling, sinks):
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        r = torch.ops.aten._scaled_dot_product_efficient_attention(
            query,
            key.expand(query.shape[0], query.shape[1], key.shape[2], key.shape[3]),
            value.expand(query.shape[0], query.shape[1], value.shape[2], value.shape[3]),
            attention_mask,
            False,
            0.0,
            False,
            scale=scaling,
        )
    out, lse_ = r[0], r[1]
    if lse_.shape[-1] != query.shape[-2]:  # xformers 布局 [B, S, H]
        lse_ = lse_.transpose(1, 2)
    # 含 sinks 的 softmax 分母 = exp(lse) + exp(sink); 输出按比例收缩
    sinks_q = sinks.reshape(1, -1, 1).expand(1, query.shape[1], query.shape[-2]).float()
    denom = torch.logaddexp(lse_.float(), sinks_q)
    factor = torch.exp(lse_.float() - denom).to(out.dtype)
    return out * factor.unsqueeze(-1)

if lse is not None:
    sinks = torch.randn(H, device=dev, dtype=bf16)
    ref = eager_ref(q, k, v, mask, D**-0.5, sinks)
    got = sdpa_sinks(q, k, v, mask, D**-0.5, sinks)
    diff = (ref.float() - got.float()).abs()
    print(f"[B] sinks-rescaled SDPA vs eager: max={diff.max():.4f} mean={diff.mean():.6f} "
          f"(ref_absmean={ref.float().abs().mean():.4f})")
    # 速度对比
    for name, fn in [("eager", lambda: eager_ref(q, k, v, mask, D**-0.5, sinks)),
                     ("sdpa+sinks", lambda: sdpa_sinks(q, k, v, mask, D**-0.5, sinks))]:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        print(f"[B] {name}: {(time.perf_counter()-t0)/10*1000:.2f} ms/fwd")

# ---------------- C. grouped_mm vs 专家循环 ----------------
E_LOCAL, T, HID, IM = 32, 1024, 4096, 2048  # EP=8: 每卡 32 专家, ~1024 token/卡
x = torch.randn(T, HID, device=dev, dtype=bf16)
w_gate_up = torch.randn(E_LOCAL, 2 * IM, HID, device=dev, dtype=bf16) * 0.02
w_down = torch.randn(E_LOCAL, HID, IM, device=dev, dtype=bf16) * 0.02
counts = torch.full((E_LOCAL,), T // E_LOCAL, dtype=torch.int32)
offs = torch.cumsum(counts, 0).to(torch.int32).to(dev)
xs = x[: int(offs[-1])]

has_gmm = hasattr(torch, "_grouped_mm")
print(f"[C] torch._grouped_mm available: {has_gmm}")
if has_gmm:
    try:
        # 权重布局: _grouped_mm 期望 [G, K, N] (转置视图)
        y = torch._grouped_mm(xs, w_gate_up.transpose(1, 2), offs=offs)
        torch.cuda.synchronize()
        print(f"[C] grouped_mm gate_up: OK out={tuple(y.shape)}")

        def loop_experts():
            outs = []
            s = 0
            for e in range(E_LOCAL):
                n = int(counts[e])
                t = xs[s : s + n]
                gu = F.linear(t, w_gate_up[e])
                g, u = gu.chunk(2, dim=-1)
                outs.append(F.linear(F.silu(g.clamp(max=10.0)) * u.clamp(min=-10.0, max=10.0), w_down[e]))
                s += n
            return torch.cat(outs)

        def gmm_experts():
            gu = torch._grouped_mm(xs, w_gate_up.transpose(1, 2), offs=offs)
            g, u = gu.chunk(2, dim=-1)
            act = F.silu(g.clamp(max=10.0)) * u.clamp(min=-10.0, max=10.0)
            return torch._grouped_mm(act, w_down.transpose(1, 2), offs=offs)

        r1, r2 = loop_experts(), gmm_experts()
        d = (r1.float() - r2.float()).abs()
        print(f"[C] grouped vs loop: max={d.max():.4f} mean={d.mean():.6f}")
        for name, fn in [("loop", loop_experts), ("grouped_mm", gmm_experts)]:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            print(f"[C] {name}: {(time.perf_counter()-t0)/20*1000:.3f} ms/fwd")
    except Exception as e:
        print(f"[C] grouped_mm failed: {type(e).__name__}: {e}")
