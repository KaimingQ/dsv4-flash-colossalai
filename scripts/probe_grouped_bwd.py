# 探查 torch._grouped_mm 在 EP grouped-experts 场景的可用性:
#   1. 反向传播 (对输入 x 的梯度; 权重冻结不需要梯度)
#   2. 不均匀分组 offs (含空组)
#   3. 与逐专家循环实现的 fwd+bwd 数值一致性
# 用法: CUDA_VISIBLE_DEVICES=0 python scripts/probe_grouped_bwd.py
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
bf16 = torch.bfloat16
E, T, H, I = 32, 1024, 4096, 2048
LIMIT = 10.0

# 不均匀 token 分布 (含 2 个空组), 模拟真实路由不均衡
counts = torch.randint(0, 60, (E,))
counts[3] = 0
counts[17] = 0
counts = (counts.float() / counts.sum() * T).round().long()
counts[0] += T - int(counts.sum())
assert counts.min() >= 0
offs = torch.cumsum(counts, 0).to(torch.int32).to(dev)
total = int(offs[-1])

w_gu = (torch.randn(E, 2 * I, H, device=dev, dtype=bf16) * 0.02).requires_grad_(False)
w_dn = (torch.randn(E, H, I, device=dev, dtype=bf16) * 0.02).requires_grad_(False)

# --- 循环参考实现 (fwd+bwd) ---
x_ref = torch.randn(total, H, device=dev, dtype=bf16, requires_grad=True)
outs = []
s = 0
cl = counts.tolist()
for e in range(E):
    n = cl[e]
    if n == 0:
        continue
    t = x_ref[s : s + n]
    gu = F.linear(t, w_gu[e])
    g, u = gu.chunk(2, dim=-1)
    outs.append(F.linear(F.silu(g.clamp(max=LIMIT)) * u.clamp(min=-LIMIT, max=LIMIT), w_dn[e]))
    s += n
y_ref = torch.cat(outs)
gout = torch.randn_like(y_ref)
y_ref.backward(gout)
dx_ref = x_ref.grad.clone()

# --- grouped_mm 实现 (fwd+bwd) ---
x_g = x_ref.detach().clone().requires_grad_(True)
try:
    gu = torch._grouped_mm(x_g, w_gu.transpose(1, 2), offs=offs)
    g, u = gu.chunk(2, dim=-1)
    act = F.silu(g.clamp(max=LIMIT)) * u.clamp(min=-LIMIT, max=LIMIT)
    y_g = torch._grouped_mm(act, w_dn.transpose(1, 2), offs=offs)
    y_g.backward(gout)
    dx_g = x_g.grad
    d_fwd = (y_ref.float() - y_g.float()).abs()
    d_bwd = (dx_ref.float() - dx_g.float()).abs()
    print(f"[grouped_mm] uneven offs + empty groups: OK total={total}")
    print(f"[grouped_mm] fwd diff: max={d_fwd.max():.4f} mean={d_fwd.mean():.6f} "
          f"(ref_absmean={y_ref.float().abs().mean():.4f})")
    print(f"[grouped_mm] bwd diff: max={d_bwd.max():.4f} mean={d_bwd.mean():.6f} "
          f"(ref_absmean={dx_ref.float().abs().mean():.4f})")
    # 权重梯度可用性 (fused 参数若 requires_grad)
    w_gu_g = w_gu.detach().clone().requires_grad_(True)
    x2 = x_ref.detach()
    gu2 = torch._grouped_mm(x2, w_gu_g.transpose(1, 2), offs=offs)
    gu2.sum().backward()
    print(f"[grouped_mm] weight grad: OK shape={tuple(w_gu_g.grad.shape)} dtype={w_gu_g.grad.dtype}")
except Exception as e:
    print(f"[grouped_mm] FAILED: {type(e).__name__}: {e}")

# --- 性能: fwd+bwd 对比 ---
import time

def loop_fb():
    x = x_ref.detach().requires_grad_(True)
    outs = []
    s = 0
    for e in range(E):
        n = cl[e]
        if n == 0:
            continue
        t = x[s : s + n]
        gu = F.linear(t, w_gu[e])
        g, u = gu.chunk(2, dim=-1)
        outs.append(F.linear(F.silu(g.clamp(max=LIMIT)) * u.clamp(min=-LIMIT, max=LIMIT), w_dn[e]))
        s += n
    y = torch.cat(outs)
    y.backward(gout)

def gmm_fb():
    x = x_ref.detach().requires_grad_(True)
    gu = torch._grouped_mm(x, w_gu.transpose(1, 2), offs=offs)
    g, u = gu.chunk(2, dim=-1)
    act = F.silu(g.clamp(max=LIMIT)) * u.clamp(min=-LIMIT, max=LIMIT)
    y = torch._grouped_mm(act, w_dn.transpose(1, 2), offs=offs)
    y.backward(gout)

for name, fn in [("loop fwd+bwd", loop_fb), ("grouped_mm fwd+bwd", gmm_fb)]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    print(f"[perf] {name}: {(time.perf_counter()-t0)/20*1000:.3f} ms")
