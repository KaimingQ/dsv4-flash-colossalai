#!/usr/bin/env python3
"""8 卡 EP 前向最小复现: 随机权重 2 层 V4ExpertsList + EpDeepseekV4MoE, 直接 GPU 初始化"""
import os
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"

from colossalai.shardformer.modeling.deepseek_v4 import (
    EpDeepseekV4MoE,
    V4ExpertsList,
)

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
cfg = AutoConfig.from_pretrained(os.path.join(MODEL_ROOT, "DeepSeek-V4-Flash-BF16-v2"), trust_remote_code=True)

class FakeMLP(torch.nn.Module):
    """模拟 SparseMoeBlock 的最小结构"""
    def __init__(self):
        super().__init__()
        self.is_hash = False
        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.top_k = cfg.num_experts_per_tok
                self.num_experts = cfg.n_routed_experts
                self.hidden_dim = cfg.hidden_size
                self.weight = torch.nn.Parameter(torch.randn(self.num_experts, self.hidden_dim, device=dev) * 0.02)
                self.routed_scaling_factor = 1.5
                import torch.nn.functional as F
                self.score_fn = lambda x: F.softmax(x, dim=-1)
            def forward(self, h):
                import torch.nn.functional as F
                flat = h.reshape(-1, self.hidden_dim)
                scores = self.score_fn(F.linear(flat, self.weight))
                idx = torch.topk(scores, self.top_k, dim=-1, sorted=False).indices
                w = scores.gather(1, idx)
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
                return None, w * self.routed_scaling_factor, idx
        self.gate = Gate()
        # experts 在外部用小规模版本赋值 (256 专家显存超单卡)
        self.experts = None
        self.shared_experts = torch.nn.Identity()

# 缩小规模: 手工构造小规模专家 (16 专家, inter=256), 避免显存爆炸
class SmallFused(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 16
        self.hidden_dim = cfg.hidden_size
        self.intermediate_dim = 256
        import torch.nn as nn
        self.act_fn = nn.SiLU()
        self.limit = 10.0
        self.gate_up_proj = torch.empty(1, dtype=torch.bfloat16)  # 仅供转换函数读 dtype

mlp = FakeMLP()
mlp.experts = V4ExpertsList(SmallFused())
mlp.gate.top_k = 4
mlp.gate.num_experts = 16
mlp.gate.weight = torch.nn.Parameter((torch.randn(16, cfg.hidden_size) * 0.02).to(device=dev, dtype=torch.bfloat16))
for e in mlp.experts.experts:
    e.gate_up_proj.data = (torch.randn(2 * 256, cfg.hidden_size) * 0.02).to(device=dev, dtype=torch.bfloat16)
    e.down_proj.data = (torch.randn(cfg.hidden_size, 256) * 0.02).to(device=dev, dtype=torch.bfloat16)
mlp = mlp.to(dev)

ep_group = dist.new_group(list(range(8)))
ep_mlp = EpDeepseekV4MoE.from_native_module(mlp, moe_dp_group=ep_group, ep_group=ep_group)

# 检查 from_native_module 后关键参数状态
_none = [n for n, p in ep_mlp.named_parameters() if p is None]
if rank == 0:
    print("None 参数样例:", _none[:6], "共", len(_none), flush=True)
    print("gate.weight:", type(ep_mlp.gate.weight).__name__, flush=True)

x = torch.randn(2, 64, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
dist.barrier()
if rank == 0:
    print("开始 EP 前向 (3 次迭代)...", flush=True)
for it in range(3):
    t0 = time.time()
    y = ep_mlp(x)
    y.float().sum().backward()
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"iter {it}: out={y.shape}, {(time.time()-t0)*1000:.0f} ms", flush=True)
print(f"rank{rank} EP dist OK", flush=True)
dist.destroy_process_group()
