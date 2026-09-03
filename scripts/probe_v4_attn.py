# v4_fast_eager_attention 与 HF 原生 eager_attention_forward 的数值对拍 + 速度对比
# 用法: CUDA_VISIBLE_DEVICES=0 python scripts/probe_v4_attn.py
import time

import torch
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mv4

from colossalai.shardformer.modeling.deepseek_v4 import v4_fast_eager_attention

torch.manual_seed(0)
dev, bf16 = "cuda", torch.bfloat16
B, H, S, D = 2, 64, 512, 512  # V4-Flash 训练形状 (seq512, bs2)


class M:
    num_key_value_groups = H  # MQA: 单 KV 头广播到 64
    training = False
    sinks = torch.randn(H, device=dev, dtype=bf16)


m = M()
q = torch.randn(B, H, S, D, device=dev, dtype=bf16)
k = torch.randn(B, 1, S, D, device=dev, dtype=bf16)
v = torch.randn(B, 1, S, D, device=dev, dtype=bf16)
causal = torch.tril(torch.ones(S, S, device=dev, dtype=torch.bool))
window = torch.arange(S, device=dev)[None, :] >= torch.arange(S, device=dev)[:, None] - 127
mask = torch.where(causal & window, 0.0, torch.finfo(bf16).min)[None].to(bf16)

ref, _ = mv4.eager_attention_forward(m, q, k, v, mask, D**-0.5, dropout=0.0)
fast, _ = v4_fast_eager_attention(m, q, k, v, mask, D**-0.5, dropout=0.0, s_aux=m.sinks, sliding_window=128)
d = (ref.float() - fast.float()).abs()
print(f"[attn] fast vs HF eager: max={d.max():.5f} mean={d.mean():.7f} "
      f"(ref_absmean={ref.float().abs().mean():.4f})")

# 反向一致性
q1 = q.detach().requires_grad_(True)
q2 = q.detach().requires_grad_(True)
r1, _ = mv4.eager_attention_forward(m, q1, k, v, mask, D**-0.5, dropout=0.0)
r2, _ = v4_fast_eager_attention(m, q2, k, v, mask, D**-0.5, dropout=0.0, s_aux=m.sinks, sliding_window=128)
g = torch.randn_like(r1)
r1.backward(g)
r2.backward(g)
db = (q1.grad.float() - q2.grad.float()).abs()
print(f"[attn] dq fast vs eager: max={db.max():.5f} mean={db.mean():.7f} "
      f"(ref_absmean={q1.grad.float().abs().mean():.4f})")

for name, fn in [
    ("HF eager", lambda: mv4.eager_attention_forward(m, q, k, v, mask, D**-0.5, dropout=0.0)),
    ("fast eager", lambda: v4_fast_eager_attention(m, q, k, v, mask, D**-0.5, dropout=0.0, s_aux=m.sinks)),
]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    print(f"[attn] {name}: {(time.perf_counter()-t0)/10*1000:.2f} ms/fwd")
