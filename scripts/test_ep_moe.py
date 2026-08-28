#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 适配单元测试:
1. get_autopolicy 能否解析 DeepseekV4ForCausalLM
2. EpDeepseekV4MoE.from_native_module 替换后, ep_size=1 下 forward 与原生实现数值一致
"""
import os

import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
dist.init_process_group(backend="gloo")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
# 仅读取 config (不加载权重)
MODEL_PATH = os.path.join(MODEL_ROOT, "DeepSeek-V4-Flash-BF16-v2")

# ---- 1. autopolicy 解析 ----
from colossalai.shardformer.policies.auto_policy import get_autopolicy  # noqa: E402

cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
cfg.num_hidden_layers = 4  # 前 3 层为 hash 路由层, 需 ≥4 层才能取到 topk 层
cfg.num_nextn_predict_layers = 0
torch.manual_seed(0)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
policy = get_autopolicy(model)
print("[1] autopolicy OK:", type(policy).__name__)

# ---- 2. EpDeepseekV4MoE 数值等价性 (ep_size=1, 随机权重, CPU) ----
from colossalai.shardformer.modeling.deepseek_v4 import (  # noqa: E402
    EpDeepseekV4MoE,
    convert_fused_experts_to_modulelist,
)

torch.manual_seed(1)
with torch.device("cpu"):
    model2 = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
# 随机初始化 (from_config 后手动填充)
for p in model2.parameters():
    if p.dtype in (torch.bfloat16, torch.float32):
        torch.nn.init.normal_(p.float() if False else p, std=0.02)

# 转换为逐专家布局 (物化权重拷贝), 再选取非 hash 路由层 (前 3 层为 hash 层)
convert_fused_experts_to_modulelist(model2)
layer = next(l for l in model2.model.layers if not l.mlp.is_hash)
native_mlp = layer.mlp

x = torch.randn(2, 16, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
with torch.no_grad():
    y_ref = native_mlp(x)

ep_group = dist.new_group([0])
ep_mlp = EpDeepseekV4MoE.from_native_module(native_mlp, moe_dp_group=ep_group, ep_group=ep_group)
with torch.no_grad():
    y_ep = ep_mlp(x)

diff = (y_ref.float() - y_ep.float()).abs().max().item()
rel = diff / (y_ref.float().abs().max().item() + 1e-9)
print(f"[2] EpDeepseekV4MoE (ep=1) vs 原生: max_abs={diff:.6e}, rel={rel:.6e}")
# bf16 下两条路径的 index_add 累加顺序不同, 舍入差异可达 1% 量级, 属正常数值噪声
assert rel < 2e-2, f"数值不等价! rel={rel}"

# ---- 3. 梯度可回传 (LoRA 训练前提) ----
x2 = torch.randn(2, 16, cfg.hidden_size, dtype=torch.bfloat16, requires_grad=True)
y = ep_mlp(x2)
y.float().sum().backward()
assert x2.grad is not None and x2.grad.abs().sum() > 0
print("[3] 反向传播 OK")

print("\nM3 单元测试全部通过")
dist.destroy_process_group()
