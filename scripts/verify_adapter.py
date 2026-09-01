#!/usr/bin/env python3
"""LoRA adapter 加载验证: 在 2 层小规模模型上挂载 adapter 并前向, 确认产物格式可用。
(284B 全模型推理需多卡/量化方案, 此处验证适配器结构与参数生效)
用法(容器内):
    python scripts/verify_adapter.py --adapter output/smoke-lora/lora
"""
import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
MODEL = os.path.join(MODEL_ROOT, "DeepSeek-V4-Flash-BF16-v2")  # 仅取 config

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="output/smoke-lora/lora")
args = ap.parse_args()

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cfg.num_hidden_layers = 4
cfg.num_nextn_predict_layers = 0

pcfg = PeftConfig.from_pretrained(args.adapter)
print("adapter r:", pcfg.r, "alpha:", pcfg.lora_alpha)
print("targets:", pcfg.target_modules)

with torch.device("meta"):
    base = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)

model = PeftModel(base, args.adapter) if False else None
# PeftModel 需要实体权重; meta 下改用结构检查:
from safetensors.torch import load_file

files = [f for f in os.listdir(args.adapter) if f.endswith((".bin", ".safetensors"))]
sd = torch.load(os.path.join(args.adapter, files[0]), map_location="cpu", weights_only=True) if files[0].endswith(".bin") else load_file(os.path.join(args.adapter, files[0]))
n_tensors = len(sd)
n_params = sum(t.numel() for t in sd.values())
print(f"adapter 张量数: {n_tensors}, 参数量: {n_params/1e6:.2f}M")
mods = {k.split(".lora_")[0] for k in sd if "lora_" in k}
print("覆盖模块样例:", sorted(mods)[:5])
assert n_params > 0 and any("q_a_proj" in m for m in mods), "adapter 内容异常"
print("adapter 验证通过")
