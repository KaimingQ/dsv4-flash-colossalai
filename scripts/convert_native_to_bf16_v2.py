#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek-V4-Flash 原生 checkpoint (FP8) -> BF16 HF 布局转换器 (v2, 可靠版)

v1 (convert_fp8_to_bf16.py) 依赖 transformers from_pretrained 自动映射, 实测发现
5.15 的映射表缺失 `hc_*_scale` 与 MTP 条目 -> 这些张量被随机初始化, 生成乱码。
本脚本不经过 transformers 加载:
  1. 反量化沿用 FlagOS 官方工具算法 (third_party/DeepSeek-V4-FlagOS/convert_weight.py)
  2. 键名按完整手工映射规则重命名 (含 hc_*_scale / sinks / indexer 等)
  3. 专家 w1/w3 拼接为融合 3D 张量 (HF DeepseekV4Experts 布局)
  4. 输出键集合与 HF meta 模型 (num_nextn_predict_layers=0) 严格校验
用法(容器内):
    # 小规模验证 (2 层)
    python scripts/convert_native_to_bf16_v2.py --num_layers 2 \
        --output $MODEL_ROOT/dsv4-flash-bf16-v2-test
    # 全量
    python scripts/convert_native_to_bf16_v2.py \
        --output $MODEL_ROOT/DeepSeek-V4-Flash-BF16-v2
"""
import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/DeepSeek-V4-FlagOS"))
from convert_weight import dequant_fp4_weight, is_expert_weight, weight_dequant  # noqa: E402

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
MODEL_PATH = os.path.join(MODEL_ROOT, "deepseek-v4-flash-0731")

RENAME = [
    # 一趟: 顶层模块名
    (".attn.", ".self_attn."),
    (".ffn.", ".mlp."),
    # 二趟: indexer 子树 (必须先于外层 compressor 规则)
    (".indexer.compressor.wgate.", ".compressor.indexer.gate_proj."),
    (".indexer.compressor.wkv.", ".compressor.indexer.kv_proj."),
    (".indexer.compressor.norm.", ".compressor.indexer.kv_norm."),
    (".indexer.compressor.ape", ".compressor.indexer.position_bias"),
    (".indexer.weights_proj.", ".compressor.indexer.scorer.weights_proj."),
    (".indexer.wq_b.", ".compressor.indexer.q_b_proj."),
    # 三趟: 外层 compressor
    (".compressor.wgate.", ".compressor.gate_proj."),
    (".compressor.wkv.", ".compressor.kv_proj."),
    (".compressor.norm.", ".compressor.kv_norm."),
    (".compressor.ape", ".compressor.position_bias"),
    # 四趟: MLA 注意力线性层
    (".wq_a.", ".q_a_proj."),
    (".wq_b.", ".q_b_proj."),
    (".wkv.", ".kv_proj."),
    (".wo_a.", ".o_a_proj."),
    (".wo_b.", ".o_b_proj."),
    (".q_norm.", ".q_a_norm."),
    (".attn_sink", ".sinks"),
    # 五趟: FFN 共享专家与 norm
    (".shared_experts.w1.", ".shared_experts.gate_proj."),
    # 路由器: 原生 gate.bias 即 HF 的 e_score_correction_bias (buffer)
    (".gate.bias", ".gate.e_score_correction_bias"),
    (".shared_experts.w2.", ".shared_experts.down_proj."),
    (".shared_experts.w3.", ".shared_experts.up_proj."),
    (".attn_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
    # 六趟: hyper-compressor 与顶层
    (".hc_attn_fn", ".attn_hc.fn"),
    (".hc_attn_base", ".attn_hc.base"),
    (".hc_attn_scale", ".attn_hc.scale"),
    (".hc_ffn_fn", ".ffn_hc.fn"),
    (".hc_ffn_base", ".ffn_hc.base"),
    (".hc_ffn_scale", ".ffn_hc.scale"),
]

EXPERT_RE = re.compile(r"^(?P<pfx>.*\.mlp)\.experts\.(?P<eid>\d+)\.(?P<w>w1|w2|w3)\.weight$")


def hf_key(native: str) -> str:
    k = native
    for src, dst in RENAME:
        k = k.replace(src, dst)
    k = re.sub(r"^embed\.weight$", "model.embed_tokens.weight", k)
    k = re.sub(r"^head\.weight$", "lm_head.weight", k)
    k = re.sub(r"^norm\.weight$", "model.norm.weight", k)
    k = re.sub(r"^hc_head_(fn|base|scale)$", r"model.hc_head.hc_\1", k)
    if k.startswith("layers."):
        k = "model." + k
    return k


def layer_of(key: str):
    m = re.match(r"(?:model\.)?layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def expected_hf_keys(model_path, num_layers):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg.num_hidden_layers = num_layers
    cfg.num_nextn_predict_layers = 0
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    keys = {n for n, _ in m.named_parameters()}
    # 缓冲区: 排除非持久化的 (如 rotary inv_freq, 运行时生成, 不存在于 checkpoint)
    for name, mod in m.named_modules():
        npb = getattr(mod, "_non_persistent_buffers_set", set())
        for bn, _ in mod.named_buffers(recurse=False):
            if bn not in npb:
                keys.add(f"{name}.{bn}" if name else bn)
    # 发布 config 的 moe_intermediate_size (2048) 与实际专家宽度 (4096) 不符,
    # 从模型定义读取真实值
    true_inter = next(mod for mod in m.modules() if mod.__class__.__name__ == "DeepseekV4Experts").intermediate_dim
    del m
    return keys, cfg, true_inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_layers", type=int, default=0, help="0 = 全量")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard_gb", type=float, default=4.5)
    args = ap.parse_args()

    idx = json.load(open(f"{MODEL_PATH}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    n_layers_total = max(layer_of(k) for k in wm if layer_of(k) is not None) + 1
    n_keep = args.num_layers or n_layers_total
    print(f"[conv] 层数: 保留 {n_keep}/{n_layers_total}, MTP 丢弃", flush=True)

    expect, cfg, true_inter = expected_hf_keys(MODEL_PATH, n_keep)
    print(f"[conv] 模型定义专家宽度: {true_inter} (config moe_intermediate_size={cfg.moe_intermediate_size})", flush=True)

    keep_key = lambda k: (  # noqa: E731
        (layer_of(k) is None or layer_of(k) < n_keep) and not k.startswith("mtp.")
    )

    # 1) 扫描: 分类张量
    scale_of = {}  # weight_key -> scale_key
    for k in wm:
        if k.endswith(".scale"):
            wk = k[: -len("scale")] + "weight"
            if wk in wm:
                scale_of[wk] = k

    sd = {}  # HF key -> tensor
    expert_part = {}  # (hf_pfx, eid) -> {w1,w2,w3}
    files = sorted(set(wm.values()))
    for fi, fn in enumerate(files):
        # 层数截断: 跳过高编号层与 mtp 分片
        fn_keys = [k for k, f in wm.items() if f == fn]
        if all((not keep_key(k)) and (layer_of(k) is not None or k.startswith("mtp.")) for k in fn_keys):
            continue
        with safe_open(f"{MODEL_PATH}/{fn}", framework="pt") as f:
            for k in fn_keys:
                if not keep_key(k):
                    continue
                if k.endswith(".scale"):
                    continue
                t = f.get_tensor(k)
                m = EXPERT_RE.match(hf_key(k))
                if m:
                    # 专家为 MXFP4 (int8 打包), 必须先反量化再收集 (否则产物为垃圾值)
                    if k in scale_of:
                        with safe_open(f"{MODEL_PATH}/{wm[scale_of[k]]}", framework="pt") as fs:
                            s = fs.get_tensor(scale_of[k])
                        t = dequant_fp4_weight(t.cuda(), s.cuda()).cpu()
                    expert_part.setdefault((m["pfx"], int(m["eid"])), {})[m["w"]] = t
                    continue
                if k in scale_of:
                    with safe_open(f"{MODEL_PATH}/{wm[scale_of[k]]}", framework="pt") as fs:
                        s = fs.get_tensor(scale_of[k])
                    # 0731 原版: 路由专家为 MXFP4 (int8 打包), 其余为 FP8; 算法均为 FlagOS 官方实现
                    if is_expert_weight(k):
                        t = dequant_fp4_weight(t.cuda(), s.cuda()).cpu()
                    else:
                        t = weight_dequant(t.cuda(), s.cuda()).cpu()
                hk = hf_key(k)
                assert hk in expect, f"映射结果不在 HF 预期键中: {k} -> {hk}"
                sd[hk] = t
        print(f"[conv] 分片 {fi+1}/{len(files)} 完成", flush=True)

    # 2) 专家融合: w1(gate)+w3(up) -> gate_up_proj [2I, H], w2 -> down_proj
    for (pfx, eid), part in expert_part.items():
        assert set(part) == {"w1", "w2", "w3"}, f"专家缺件: {pfx}.{eid}"
        sd[f"{pfx}.experts.experts.{eid}.gate_up_proj"] = torch.cat([part["w1"], part["w3"]], dim=0)
        sd[f"{pfx}.experts.experts.{eid}.down_proj"] = part["w2"]

    # 3) 键校验 (逐专家布局: 预期融合键 -> 展开为逐专家键)
    expect_pe = set()
    for k in expect:
        m = re.match(r"^(?P<pfx>.*)\.experts\.(gate_up_proj|down_proj)$", k)
        if m:
            if k.endswith("gate_up_proj"):
                for e in range(cfg.n_routed_experts):
                    expect_pe.add(f"{m['pfx']}.experts.experts.{e}.gate_up_proj")
                    expect_pe.add(f"{m['pfx']}.experts.experts.{e}.down_proj")
            # 融合键本身不保留 (已展开为逐专家键)
        else:
            expect_pe.add(k)
    missing = sorted(expect_pe - set(sd))[:8]
    unexpected = sorted(set(sd) - expect_pe)[:8]
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    print(f"[conv] 键校验通过: {len(sd)} 键 (逐专家布局)", flush=True)

    # 4) 分片保存
    os.makedirs(args.output, exist_ok=True)
    max_bytes = int(args.shard_gb * 1024**3)
    keys_sorted = sorted(sd)
    bounds, cur, start = [], 0, 0
    for i, k in enumerate(keys_sorted):
        nb = sd[k].numel() * sd[k].element_size()
        if cur + nb > max_bytes and i > start:
            bounds.append(start)
            start, cur = i, 0
        cur += nb
    bounds.append(start)
    total = len(bounds)
    weight_map = {}
    for si, s0 in enumerate(bounds):
        s1 = bounds[si + 1] if si + 1 < total else len(keys_sorted)
        shard = {k: sd[k].contiguous() for k in keys_sorted[s0:s1]}
        out_fn = f"model-{si+1:05d}-of-{total:05d}.safetensors"
        save_file(shard, os.path.join(args.output, out_fn), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = out_fn
            del sd[k]
        print(f"[conv] 保存 {out_fn} ({len(shard)} 键)", flush=True)
    json.dump({"metadata": {"total_size": None}, "weight_map": weight_map},
              open(os.path.join(args.output, "model.safetensors.index.json"), "w"), indent=1)

    # 5) config / tokenizer (去 MTP 与量化字段)
    cfg_dict = cfg.to_dict()
    cfg_dict.pop("quantization_config", None)
    cfg_dict.pop("expert_dtype", None)
    cfg_dict["num_nextn_predict_layers"] = 0
    cfg_dict["moe_intermediate_size"] = true_inter
    cfg_dict["torch_dtype"] = "bfloat16"
    json.dump(cfg_dict, open(os.path.join(args.output, "config.json"), "w"), indent=2, ensure_ascii=False)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True).save_pretrained(args.output)
    print(f"[conv] 完成 -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
