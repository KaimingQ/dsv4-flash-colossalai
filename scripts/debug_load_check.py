#!/usr/bin/env python3
"""权重加载完整性校验: 复刻训练脚本构建流 (lazy init + 专家拆分 + LoRA + boost + load_model),
逐参数与 checkpoint 比对, 找出未被加载 (保持随机初始化) 的参数。
用法(容器内): colossalai run --nproc_per_node 8 scripts/debug_load_check.py
"""
import json
import os

import torch
import torch.distributed as dist
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import MoeHybridParallelPlugin
from colossalai.lazy import LazyInitContext
from colossalai.utils import get_current_device

MODEL = os.environ.get("MODEL", "/home/shared/deepseek-ai/DeepSeek-V4-Flash-BF16-v2")
FP32_STRICT = os.environ.get("FP32_STRICT", "")  # 复现 HF 的 _keep_in_fp32_modules_strict

colossalai.launch_from_torch()
rank = dist.get_rank()
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)

plugin = MoeHybridParallelPlugin(
    ep_size=8, tp_size=1, pp_size=1, zero_stage=1, cpu_offload=False,
    sp_size=1, sequence_parallelism_mode="split_gather", enable_sequence_parallelism=False,
    enable_fused_normalization=False, enable_flash_attention=False,
    max_norm=1.0, precision="bf16", microbatch_size=1,
)
booster = Booster(plugin=plugin)

# 预热导入模型模块 (同 lora_finetune.py): 懒初始化内首次导入会因注解求值崩溃
_ = AutoModelForCausalLM._model_mapping[type(cfg)]

with LazyInitContext(default_device=get_current_device()):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, attn_implementation="eager", torch_dtype=torch.bfloat16)
    from colossalai.shardformer.modeling.deepseek_v4 import convert_fused_experts_to_modulelist
    convert_fused_experts_to_modulelist(model)

    # 同 lora_finetune.py: 重建 RotaryEmbedding 懒链 buffer
    from colossalai.lazy.lazy_init import LazyTensor
    for _, mod in model.named_modules():
        if not mod.__class__.__name__.endswith("RotaryEmbedding"):
            continue
        lazy_bufs = [n for n, b in mod.named_buffers() if isinstance(b, LazyTensor)]
        if not lazy_bufs:
            continue
        with torch.device("cpu"):
            fresh = mod.__class__(mod.config)
        for n in lazy_bufs:
            v = getattr(fresh, n, None)
            if v is not None:
                mod.register_buffer(n, v, persistent=False)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32,
        target_modules=["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = booster.enable_lora(model, lora_config=lora_config)

    # 复现 HF from_pretrained 的 _keep_in_fp32_modules_strict: 这些模块保持 fp32
    if FP32_STRICT:
        _fp32_kw = ("attn_hc", "ffn_hc", "hc_head", "sinks", "position_bias", "e_score_correction_bias",
                    "q_a_norm", "kv_norm", "input_layernorm", "post_attention_layernorm")
        for n, m in model.named_modules():
            leaf = n.split(".")[-1]
            if leaf in _fp32_kw or n.endswith("model.norm"):
                m.float()
model.train()
model.gradient_checkpointing_enable()

from colossalai.nn.optimizer import HybridAdam
optimizer = HybridAdam(model_params=model.parameters(), lr=2e-5)
model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
booster.load_model(model, MODEL, low_cpu_mem_mode=True, num_threads=8)

# ---- 与 checkpoint 比对 (本 rank 持有的参数+buffer) ----
from safetensors import safe_open
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]

# 构建 模型参数名(去 lora 前缀) -> checkpoint 键 的映射: 直接按同名找
named = [(n, p) for n, p in model.named_parameters() if p is not None and "lora_" not in n]
named += [(n, b) for n, b in model.named_buffers() if b is not None and "inv_freq" not in n]
# 尝试剥离包装前缀 (colossalai module. + PEFT base_model. + base_layer)
def ckpt_key(name: str):
    name = name.replace(".base_layer.", ".")
    changed = True
    while changed:
        changed = False
        for pref in ("module.", "base_model.model.", "base_model."):
            if name.startswith(pref):
                name = name[len(pref):]
                changed = True
    return name

import collections
status = collections.defaultdict(list)
files_open = {}
checked = 0
mismatch = 0
for n, p in named:
    ck = ckpt_key(n)
    if ck not in idx:
        status["not_in_ckpt"].append(n)
        continue
    fn = idx[ck]
    if fn not in files_open:
        files_open[fn] = safe_open(os.path.join(MODEL, fn), framework="pt", device="cpu")
    ref = files_open[fn].get_tensor(ck)
    cur = p.detach().to("cpu", dtype=ref.dtype if ref.dtype != torch.float32 else torch.float32)
    if cur.shape != ref.shape:
        status["shape_mismatch"].append((n, tuple(cur.shape), tuple(ref.shape)))
        continue
    d = (cur.float() - ref.float()).abs().max().item()
    checked += 1
    if d > 1e-3:
        mismatch += 1
        status["diff"].append((n, round(d, 5)))

for f in files_open.values():
    pass
print(
    f"[load-check rank{rank}] 比对 {checked} 个, 不一致 {mismatch} 个, "
    f"无ckpt键 {len(status['not_in_ckpt'])} 样例 {status['not_in_ckpt'][:4]}",
    flush=True,
)
if status["shape_mismatch"]:
    print(f"[load-check rank{rank}] shape 不符: {status['shape_mismatch'][:5]}", flush=True)
if status["diff"]:
    print(f"[load-check rank{rank}] 数值差异样例: {status['diff'][:10]}", flush=True)

# ---- 前向探针: 训练路径模型在真实样本上的 loss (全部 rank 相同输入, 与训练同模式) ----
import json as _json
from coati.dataset.loader import apply_chat_template_and_mask
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
rows = [_json.loads(l) for l in open("data/sft_smoke.jsonl")][:2]
with torch.no_grad():
    for i, r in enumerate(rows):
        t = apply_chat_template_and_mask(tok, r, 512, "")
        batch = {k: v.unsqueeze(0).to(get_current_device()) for k, v in t.items()}
        out = model(**batch)
        loss_r = out.loss.detach().clone()
        dist.all_reduce(loss_r)
        if rank == 0:
            print(f"[fwd-probe] sample{i} loss={(loss_r/dist.get_world_size()).item():.4f}", flush=True)
dist.barrier()
dist.destroy_process_group()
