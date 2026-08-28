#!/usr/bin/env bash
# ORPO 后训练 (容器内执行): 无参考模型的单阶段偏好优化 (SFT loss + odds-ratio 偏好项)
# 284B 显存策略: moe 插件 (EP=8 专家切分 + ZeRO1) + LoRA + 单模型 (无 ref, 免第二份 530GB)
# 数据: 与 DPO 共享数学偏好对 (data/dpo_public)
# 攻坚记录: 初版 ~14 步 nan 的根因链已定位并修复 (见补丁):
#   1) OddsRatioLoss 的 log(sigmoid(Δ)) 在 Δ 极负时 sigmoid 下溢为 0 -> log(0)=-inf;
#   2) masked 位置 logp=0, log1p(-exp(0))=-inf, 乘 mask 得 -inf*0=nan (每步必触发),
#      修复: torch.where 选分支在反向仍漏 -inf 梯度, 改用 logp.clamp(max=-1e-4), 有效且梯度有界;
#   3) trainer 把模型内 bf16 CE 作为 chosen NLL, 大 logits 溢出为 nan;
#      修复: 不传 labels, 与 SimPO 路线一致由 fp32 logits 重算 CE;
#   4) set_detect_anomaly(True) 每步拖慢一倍, 一并移除;
#   5) save_interval=600 > 每 rank 的 micro 步数 (532), 即训练中不做中间检查点,
#      仅脚本末尾保存最终产物: 每 50 优化步写一次 530GB EP 分片在 NFS 上实测崩溃。
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# 逐专家键布局 (EP 按需加载专家)
MODEL=${MODEL:-"${MODEL_ROOT:?需先设置 MODEL_ROOT(模型权重所在目录)}/DeepSeek-V4-Flash-BF16-v2"}
SAVE_DIR=${SAVE_DIR:-output/dsv4-orpo}

TRAIN_SCRIPT=third_party/ColossalAI/applications/ColossalChat/examples/training_scripts/train_orpo.py

colossalai run --nproc_per_node 8 "${TRAIN_SCRIPT}" \
    --pretrain "${MODEL}" \
    --dataset data/dpo_public \
    --plugin moe \
    --ep 8 \
    --zero_stage 1 \
    --lora_config configs/lora_dpo.json \
    --mixed_precision bf16 \
    --lr 5e-6 \
    --lam 0.1 \
    --max_length 512 \
    --batch_size 1 \
    --accumulation_steps 4 \
    --max_epochs 1 \
    --grad_clip 1.0 \
    --grad_checkpoint \
    --save_interval 600 \
    --save_dir "${SAVE_DIR}" \
    --config_file output/orpo_config.json 2>&1 | tee logs/orpo_run.log
