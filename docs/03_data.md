# 03 数据准备（统一数学领域）

> 设计原则：SFT / DPO(SimPO) / ORPO 全部使用**同一领域（数学推理）**数据，
> 保证训练效果有针对性。

## 数据源

| 阶段 | 数据集 | 用途 |
|---|---|---|
| SFT | AI-MO/NuminaMath-CoT | 数学题 + CoT 解答，指令微调 |
| DPO / ORPO | NuminaMath 多解质量偏好对（同题不同解，按完整度/格式构造 chosen/rejected） | 偏好对齐 |

## 格式要求

- SFT（`lora_finetune.py` → `RawConversationDataset`）：每行一个消息数组
  ```json
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
  ```
- DPO/ORPO（`train_dpo.py`/`train_orpo.py` → `load_tokenized_dataset`）：arrow 格式，
  含 `chosen_input_ids`/`chosen_loss_mask`/`rejected_input_ids`/`rejected_loss_mask`

## 使用

```bash
docker exec dsv4-colossal-train bash -c "bash scripts/prepare_data.sh"
# 数据量由 MAX_SAMPLES 控制, 例如正式训练:
docker exec dsv4-colossal-train bash -c "MAX_SAMPLES=10000 bash scripts/prepare_data.sh"
```

## 实现要点与踩坑

1. `datasets` streaming 长迭代在本环境会偶发进程退出时
   `terminate called without an active exception`（abort，rc=-6），**文件已完整写出**；
   转换器改为分批子进程模式（每批 500 条，`--skip` 偏移续拉），按文件内容判定成败。
2. `HF_DATASETS_CACHE` 必须指向本地盘（`/tmp/hf_cache`）：NFS 上 mmap 会 core dump。
3. HF 镜像端点与代理不可同时使用，二者择一。
4. 样本长度过滤（题+答 ≤ 3000 字符），与训练 `max_length` 匹配，减少无效截断。
