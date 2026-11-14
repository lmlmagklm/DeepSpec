# DSparkX 设计文档：共享表征 → 轮廓专家 → K 小专家 → 逐位裁决 → 线索回补

DSparkX 是在 [DeepSpec](../README.md) 的 DSpark 草稿模型基础上的改进，对应以下解码流水线
（与原始设计图一一对应）：

```
共享向量表征 H ──► 通用专家·出轮廓 ──► 小专家×K 并行解码 ──► 逐位裁决：该不该上？
                                              │                    │
                                              │              上 ──► 落定 token · 下一位
                                              │                    │
                                              │              不上 ─► 不淘汰·补前面结构
                                              ▼                    （候选留作线索，补齐前缀）
                                       各出候选 + 置信度 ──────────────► 再上
```

## 逐段对应

### 1. 共享向量表征 H

沿用 DSpark：目标模型多层隐状态经 `fc` + `hidden_norm` 融合成共享表征
（`deepspec/modeling/dspark/common.py::extract_context_feature`，
`Qwen3DSparkModel._forward_backbone`）。

### 2. 通用专家 · 出轮廓（新增）

`deepspec/modeling/dsparkx/skeleton.py::SkeletonExpert`

- 对骨干隐状态做低秩塑形（RMSNorm → Linear(d→skeleton_dim) → GELU）；
- 输出两类信号：
  - **sketch**（草图向量，回投隐空间）：作为全部小专家的公共条件输入，
    保证 K 个专家在同一轮廓上分工细化；
  - **coarse logits**（低秩词表投影的粗分布）：块级「轮廓」，
    由 `skeleton_ce_alpha` 加权监督。

### 3. 小专家 × K 并行解码，各出候选 + 置信度（新增）

`deepspec/modeling/dsparkx/experts.py::SmallExpertPool`

- 每个小专家 = 低秩残差分支：输入 `[H; sketch; clue?]` 经窄 hidden MLP
  产出隐空间残差 delta，门控 sigmoid 尺度化后加回共享隐状态，再经
  **共享 lm_head** 得到候选分布。参数开销为 `K × (d+s+c) × h` 量级，
  不复制 K 份全词表投影——保持「小专家」；
- 每个专家自带置信度头（`AcceptRatePredictor`），输出该位置候选的预期
  接受率 logit；
- 候选融合（`common.py::merge_expert_probs`）：按置信度 sigmoid 归一化
  为混合权重，对各专家分布做凸组合，得到提案分布。

### 4. 逐位裁决：该不该上？（沿用）

DeepSpec 的 rejection sampling 验证完全不变
（`deepspec/eval/base_evaluator.py::verify_draft_tokens`）：目标模型逐位
核验草稿 token，接受前缀，首个分歧处按残差分布采样修正 token。

### 5. 不上 → 不淘汰 · 补前面结构，候选留作线索，补齐前缀（新增）

- 推理侧（`deepspec/eval/dsparkx/draft_ops.py::extract_rejection_clues`）：
  裁决在块内停止时，被拒绝的候选后缀**不丢弃**，作为「线索」存入上下文；
  下一轮提案把线索 token 填入块前缀位置（补齐前面结构），K 个专家在
  补齐后的结构上继续解码剩余位置，然后「再上」（重新送目标模型裁决）。
  完全通过（accepted == proposed）时清空线索。
- 训练侧（`Qwen3DSparkXModel.forward` 的 recovery pass）：把合并分布的
  argmax 当作「模拟被拒绝的候选」线索，做一轮线索条件下的回补前向，
  用 `clue_recovery_alpha` 加权的 CE 监督真实 token——即训练
  「拿着错误线索也能补齐前缀、回归正确结构」的能力。
  监控指标：`clue_fix`（回补修正命中）/ `clue_break`（回补引入错误）。

## 训练损失

```
loss = ce_loss_alpha        · CE(合并提案分布)
     + expert_ce_alpha      · mean_k CE(专家k)          # 默认 0.3
     + skeleton_ce_alpha    · CE(轮廓粗分布)            # 默认 0.1
     + l1_loss_alpha        · L1(合并分布, 目标分布)
     + confidence_head_alpha· BCE(专家置信度, 专家接受率)
     + clue_recovery_alpha  · CE(线索回补分布)          # 默认 0.3
```

所有项共享 `loss_weight_mask`（eval_mask × 可选指数位置衰减）。

## 文件清单

| 路径 | 作用 |
| --- | --- |
| `deepspec/modeling/dsparkx/common.py` | 输出结构、置信度混合与融合 |
| `deepspec/modeling/dsparkx/skeleton.py` | 通用轮廓专家 |
| `deepspec/modeling/dsparkx/experts.py` | K 小专家池（含置信度头、线索条件） |
| `deepspec/modeling/dsparkx/qwen3/modeling.py` | Qwen3DSparkXModel：训练前向 + 推理提案 |
| `deepspec/modeling/dsparkx/qwen3/config.py` | draft config 构建 |
| `deepspec/modeling/dsparkx/loss.py` | 六分项训练损失 |
| `deepspec/trainer/dsparkx_trainer.py` | 训练器 |
| `deepspec/eval/dsparkx/draft_ops.py` | 推理提案与线索提取 |
| `deepspec/eval/dsparkx/evaluator.py` | 裁决-回补闭环评测器 |
| `config/dsparkx/dsparkx_qwen3_4b.py` | Qwen3-4B 训练配置 |
| `tests/test_dsparkx_smoke.py` | 形状与梯度冒烟测试 |

## 使用

训练（与 DeepSpec 一致，仅换 config）：

```bash
bash scripts/train/train.sh  # config_path=config/dsparkx/dsparkx_qwen3_4b.py
```

评测（草稿 checkpoint 的 `architectures` 为 `Qwen3DSparkXModel` 时自动路由）：

```bash
python eval.py --target_name_or_path <target> --draft_name_or_path <dsparkx_ckpt> \
    --max-new-tokens 2048 --confidence-threshold 0.6
```
