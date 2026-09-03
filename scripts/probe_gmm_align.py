# 确定 torch._grouped_mm 的对齐/空组约束 (H20, bf16):
#   逐级测试: 均匀组 fwd / 均匀组 bwd / 8 倍数不均匀 bwd / 任意不均匀 fwd / 空组 fwd
# 用法: CUDA_VISIBLE_DEVICES=0 python scripts/probe_gmm_align.py
import torch

torch.manual_seed(0)
dev, bf16 = "cuda", torch.bfloat16
E, H, I = 8, 4096, 2048  # 小组数加快迭代


def make(counts_list):
    counts = torch.tensor(counts_list, dtype=torch.int32)
    offs = torch.cumsum(counts, 0).to(torch.int32).to(dev)
    total = int(counts.sum())
    x = torch.randn(total, H, device=dev, dtype=bf16, requires_grad=True)
    w = torch.randn(E, 2 * I, H, device=dev, dtype=bf16) * 0.02
    return x, w, offs, total


def case(name, counts_list, do_bwd):
    try:
        x, w, offs, total = make(counts_list)
        y = torch._grouped_mm(x, w.transpose(1, 2), offs=offs)
        if do_bwd:
            y.backward(torch.randn_like(y))
        torch.cuda.synchronize()
        print(f"[{name}] OK")
        return True
    except Exception as e:
        torch.cuda.synchronize() if False else None
        print(f"[{name}] FAIL: {type(e).__name__}: {str(e)[:120]}")
        return False


import subprocess, sys

cases = [
    ("uniform-32 fwd", [32] * E, False),
    ("uniform-32 bwd", [32] * E, True),
    ("mult8 uneven bwd", [8, 16, 24, 32, 40, 48, 56, 64], True),
    ("arbitrary fwd", [31, 17, 25, 33, 41, 47, 55, 63], False),
    ("empty-group fwd", [32, 0, 32, 32, 32, 32, 32, 32], False),
    ("empty-group bwd", [32, 0, 32, 32, 32, 32, 32, 32], True),
]
# 每个 case 用子进程隔离 (device-side assert 会污染 CUDA context)
if len(sys.argv) > 1:
    i = int(sys.argv[1])
    case(*cases[i])
else:
    for i, (name, c, b) in enumerate(cases):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip().split("\n")
        line = next((l for l in out if l.startswith("[")), out[-1] if out else "?")
        print(line[:160])
