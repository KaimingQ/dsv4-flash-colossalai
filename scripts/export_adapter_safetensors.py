#!/usr/bin/env python3
"""将 adapter_model.bin 导出为 adapter_model.safetensors (验收产物格式)"""
import sys

import torch
from safetensors.torch import save_file

src = sys.argv[1] if len(sys.argv) > 1 else "output/dsv4-lora-sft-full/lora/adapter_model.bin"
dst = src.replace(".bin", ".safetensors")
sd = torch.load(src, map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, dst, metadata={"format": "pt"})
print(f"saved {dst} ({len(sd)} tensors)")
