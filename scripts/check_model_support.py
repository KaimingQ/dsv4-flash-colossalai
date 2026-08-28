#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 验证: transformers 对 deepseek_v4 的原生支持检查
1. 注册表中是否存在 deepseek_v4 model_type
2. AutoConfig / AutoTokenizer 能否加载真实模型目录
3. 权重格式 (FP8) 是否被加载器识别
不加载全部权重, 仅做结构级验证。
"""
import os
import sys

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
MODEL_PATH = os.path.join(MODEL_ROOT, "deepseek-v4-flash-0731")

print("=" * 60)
print("1) transformers 注册表检查")
from transformers import AutoConfig, AutoTokenizer

try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    print("  已注册的 deepseek* model_type:", [k for k in CONFIG_MAPPING_NAMES if "deepseek" in k])
except Exception as e:
    print("  检查失败:", e)

print("=" * 60)
print("2) AutoConfig.from_pretrained")
try:
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("  OK, model_type =", cfg.model_type, ", arch =", cfg.architectures)
    print("  num_hidden_layers =", cfg.num_hidden_layers, ", hidden_size =", cfg.hidden_size)
    print("  quantization_config =", getattr(cfg, "quantization_config", None))
except Exception as e:
    print("  失败:", type(e).__name__, e)
    sys.exit(1)

print("=" * 60)
print("3) AutoTokenizer.from_pretrained")
try:
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ids = tok("hello 你好", return_tensors="pt")["input_ids"]
    print("  OK, vocab_size =", tok.vocab_size, ", encode 样例 =", ids.shape)
except Exception as e:
    print("  失败:", type(e).__name__, e)

print("=" * 60)
print("4) 模型类是否可实例化 (from_config, meta device, 不占显存)")
try:
    import torch
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK, class = {type(model).__name__}, params = {n_params/1e9:.2f}B")
    # 打印模块名, 供 LoRA target 与 shardformer 适配参考
    names = [n for n, _ in model.named_modules()]
    sample = [n for n in names if "layers.0." in n][:20]
    print("  layer0 模块名样例:")
    for n in sample:
        print("   ", n)
except Exception as e:
    print("  失败:", type(e).__name__, e)
    sys.exit(2)

print("=" * 60)
print("M2 结构级验证完成")
