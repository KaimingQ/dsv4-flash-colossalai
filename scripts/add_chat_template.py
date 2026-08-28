#!/usr/bin/env python3
"""为转换产物补充 DeepSeek 风格 chat_template (原生发布版未提供)。
用法: python scripts/add_chat_template.py <产物目录> [<产物目录2> ...]
"""
import json
import os
import sys

TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] }}{% endif %}"
    "{% if message['role'] == 'user' %}<｜User｜>{{ message['content'] }}{% endif %}"
    "{% if message['role'] == 'assistant' %}<｜Assistant｜>{{ message['content'] }}"
    "<｜end▁of▁sentence｜>{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<｜Assistant｜>{% endif %}"
)

MODEL_ROOT = os.environ.get("MODEL_ROOT")
if MODEL_ROOT is None:
    raise SystemExit("需设置环境变量 MODEL_ROOT (模型权重所在目录)")
paths = sys.argv[1:] or [os.path.join(MODEL_ROOT, "DeepSeek-V4-Flash-BF16-v2")]
paths = [p if p.endswith("tokenizer_config.json") else f"{p.rstrip('/')}/tokenizer_config.json" for p in paths]
for path in paths:
    c = json.load(open(path))
    c["chat_template"] = TEMPLATE
    json.dump(c, open(path, "w"), indent=2, ensure_ascii=False)
    print("patched", path)
