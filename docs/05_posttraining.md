# 05 后训练：DPO/SimPO 与 ORPO（284B @ 8×H20）

> 统一数学领域偏好数据（NuminaMath 多解质量构造，4257 对）上的两种偏好优化后训练，
> 均在单模型显存预算内完整跑通并稳定收敛。
> ——build high-quality private models at low cost。

## 1. DPO / SimPO（`scripts/run_dpo.sh`）

| 项 | 值 |
|---|---|
| 插件 | `--plugin moe`（本项目为 `train_dpo.py` 新增的 EP 路线） |
| 损失 | SimPO（`--disable_reference_model`，省去第二份 568GB 参考模型） |
| 超参 | beta=2.0, gamma=0.5, lr=5e-6, seq512, bs2, acc2, LoRA r16 |
| 数据 | 4257 偏好对 × 1 epoch（133 步，31 分钟，14.2 s/it） |
| 结果 | loss 全程稳定在 ~1.0（0.90~1.10），reward accuracy ~0.47（chosen/rejected 区分有限） |
| 产物 | `output/dsv4-dpo/modeling/`（EP 分片，`merge_ep_shards_to_hf.py` 可导出 HF 格式） |

插件选型记录（如实）：
- `gemini`：GeminiPlugin 显式不支持 LoRA → 排除；
- `3d`（HybridParallel）：无专家并行切分，融合专家 568GB 落单卡 → OOM 实测；
- `moe`（MoeHybridParallelPlugin）：EP=8 专家切分 + ZeRO1，与 SFT 同一验证路线 → 通过。

## 2. ORPO（`scripts/run_orpo.sh`）——nan 攻坚全记录

| 项 | 值 |
|---|---|
| 超参 | lam=0.1, lr=5e-6, seq512, bs1×acc4（有效批 32）, LoRA r16, grad_checkpoint |
| 数据 | 与 DPO 共享 4257 偏好对 × 1 epoch（133 优化步，40 分钟，21.9 s/it） |
| 结果 | loss ~1.1 → **~0.5** 稳定下降，**全程零 nan** |
| 显存 | 峰值 **84.7GB/卡**（96GB 的 88%）；bs2 时 97GB 顶满死锁，故取 bs1×acc4 |
| 产物 | `output/dsv4-orpo/modeling/`（626GB EP 分片，可导出 HF 格式） |

### 2.1 初版失败现象

初版训练 ~14 步出现 nan 梯度（`LogBackward0 returned nan values`，lr 5e-6/1e-6 均复现）。
逐层定位后确认为**三层叠加的数值问题**（均在上游官方代码中，非本项目适配引入）：

### 2.2 根因链与修复（补丁见 `patches/0001`）

| # | 根因 | 触发机制 | 修复 |
|---|---|---|---|
| 1 | `OddsRatioLoss` 用 `log(sigmoid(Δ))` | Δ 极负（284B 高置信数学解答常见）时 sigmoid 在 fp32 下溢为 0 → `log(0) = -inf` | 改用 `-softplus(-Δ)`（= `log σ(Δ)` 的稳定形式），恒有限、梯度 ∈(0,1) |
| 2 | `1.0001` epsilon hack | `log(-exp(logp)+1.0001)` 在 logp→0 处梯度 `1/(1.0001-exp)` ≈ 1e4 尖峰 | 改用 `log1p(-exp(logp))` 精确形式 |
| 3 | **masked 位置 logp=0**（最致命） | `calc_masked_log_probs` 把 mask 乘进 logp，被 mask 位置恰为 0 → `log1p(-1) = -inf` → `-inf×0 = nan`，**每步必触发**；`torch.where` 选分支在反向仍漏 -inf 梯度 | `logp.clamp(max=-1e-4)`：仅影响近概率 1 的 token，梯度有界（~1e4，由 grad_clip 兜底） |
| 4 | trainer 直接取模型内 **bf16 交叉熵**作为 chosen NLL | 大 logits 下 bf16 CE 溢出，实测第 1 步即 nan（诊断探针证实：logits/nll fp32 均有限前 or_loss 已 nan） | 不传 `labels`，与 SimPO 路线一致由 **fp32 logits 重算 CE** |
| 5 | `set_detect_anomaly(True)` 每步开启 | 反向慢一倍，且是 nan 警告的来源（干扰定位） | 移除 |
| 6 | `--save_interval 200` | 每 50 优化步写一次 530GB EP 分片，NFS 上 rank4 文件打开失败崩溃 | `--save_interval 600`（>532 micro 步），仅末尾保存 |

诊断手段（可复用）：在损失合成处插入一次性探针，打印 `logits/logp/nll/or_loss` 各项
有限性，10 分钟内锁定 nan 所在分支（`or_loss=nan, lor=nan, nll 有限`）。

## 3. 后训练算法选型说明（如实记录）

- 284B 在 8×96GB 上，**带参考模型的算法**（如需要 actor+ref 双模型各 ~84GB/卡的
  RL 方案）超出单卡容量，不具可行性；
- 因此本项目选择**无参考模型**的偏好优化路线：SimPO（DPO 的无参考变体）与
  ORPO，两者均可在单模型显存预算内完成，且数据格式互通；
- ORPO 初版的数值不稳定**并非算法本身缺陷**，而是上游实现的三处数值漏洞；
  修复后稳定收敛，两者共同构成 284B 显存受限下的完整后训练工具箱。

## 4. 产物导出

训练产物为 EP 分片格式，可用 `merge_ep_shards_to_hf.py` 合并导出为
HuggingFace 融合布局（逐专家键 → 3D 专家张量，含键集合严格校验），
供下游推理服务加载：

```bash
docker exec dsv4-colossal-train bash -c "cd /workspace && python scripts/merge_ep_shards_to_hf.py \
    --input output/dsv4-dpo/modeling --output output/dsv4-dpo-hf"
docker exec dsv4-colossal-train python scripts/add_chat_template.py output/dsv4-dpo-hf
```

各阶段训练数据（步数/吞吐/显存/收敛）见 `docs/06_report.md`。

## 3. 吞吐优化后的端到端数据

同步消除 + grouped GEMM + attention 提速后（loss 与基线一致，见 `docs/07_improvements.md` 第 10 节）：

| 阶段 | 优化前 | 优化后 |
|---|---|---|
| DPO/SimPO（133 步全量重训） | 14.2 s/it（31 分钟） | **11.26 s/it（25 分 13 秒）**，loss 终值 0.997 |
| ORPO（133 步全量重训） | 21.9 s/it（40 分钟） | **13.12 s/it（29 分 04 秒）**，loss 末值 0.70，零 nan |
| LoRA SFT（1250 步） | 4.14 s/it（约 86 分钟） | **2.98 s/it（约 62 分钟）**，loss 终值 0.383 |
