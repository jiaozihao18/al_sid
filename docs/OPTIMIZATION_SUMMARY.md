# 代码创新与优化点总结

本文档汇总 al_sid 项目中 **algr**（语义 ID 语言模型训练）与 **SID_generation**（RQ-VAE 码本训练）的主要创新点与工程优化。

---

## 一、语言模型训练（algr）

### 1.1 核心创新

#### LM Head 优化

- **背景**：SID 序列预测场景下 prompt 长、answer 短，全序列计算 logits 显存占用高，大词表时尤为明显。
- **方案**：训练时仅对有效 label 位置计算 logits，通过 `min_first_non_neg_index` 定位首个非负 label，再以 `slice_indices` 切片。
- **效果**：显著降低 lm_head 计算量与显存占用。

#### 动态 Beam Search

- **背景**：SID 由 3 层 code 组成（`max_new_tokens=3`），标准 Beam Search 固定宽度，候选多样性有限。
- **方案**：自定义 `_beam_search`，在每个生成步骤逐步扩大 beam 宽度，最大化候选空间。
- **实现细节**：
  1. **初始 `beams_to_keep` 计算**：`max(2, 1 + n_eos_tokens) × num_beams`。Qwen2.5 有 2 个 EOS token（`<|endoftext|>` 和 `<|im_end|>`），故 `beams_to_keep = 3 × num_beams`。
  2. **动态扩展调度表**（`num_return_sequences == num_beams` 时）：

     | 生成步骤 | num_beams | beams_to_keep | 示例（num_beams=100） |
     |----------|-----------|---------------|----------------------|
     | Step 1   | B         | 3B            | 100 → 保留 300       |
     | Step 2   | 2B        | 6B            | 200 → 保留 600       |
     | Step 3   | 4B        | **12B**       | 400 → 保留 **1200**  |

  3. **`top_num_beam_mask` 全量保留**：原始 HF 实现仅保留 top `num_beams` 个完成序列，其余 mask 为 False；本实现将 mask 改为全 True（`torch.ones((beams_to_keep))`），使所有 `beams_to_keep` 个候选均参与最终输出。
  4. **输出不截断**：`_beam_search` 返回时直接 `_flatten_beam_dim(sequences)` 输出全部候选，不按 `num_return_sequences` 裁剪。
- **最终效果**：配置 `num_beams=100, max_new_tokens=3` 时，每个 item 实际返回 `4 × 3 × 100 = 1200` 个候选序列，远超标准 Beam Search 的 100 个，显著提升召回多样性。
- **两种模式**：`num_return_sequences == num_beams`（离线批量推理）与 `!=`（在线推理，扩展系数略有不同）。

### 1.2 训练任务

| 任务 | 说明 | 配置 |
|------|------|------|
| **Pretrain** | 预训练，input 直接作为文本，无 instruction 格式 | `training_mode: "pretrain"` |
| **SFT** | 监督微调，采用 instruction 格式（system/user/assistant） | `training_mode` 非 pretrain |

### 1.3 模型加载方式

| 方式 | 说明 | 配置 |
|------|------|------|
| **dense** | 加载预训练，随机初始化 embedding，freeze 其余层，仅训练 embedding | `load_func: "dense"` |
| **scratch** | 从 config 初始化空模型，随机权重 | `load_func: "scratch"` |
| **from pretrain** | 全量加载预训练，全量微调 | `load_func` 不配置或非 scratch/dense |

### 1.4 工程与基础设施

- **训练与推理一体化**：同一 Runner 支持 `do_train` / `do_eval` / `do_predict`，通过配置切换即可完成训练与推理。
- **多 NPU 并行**：支持 `device_type: npu`，采用 hccl backend，与 CUDA 统一设备抽象。
- **Dataset Streaming**：支持 `streaming=True` 流式加载大规模数据集。
- **预测输出**：PredictWriter 分布式分片写入、DataCollatorWrapper 保留非模型字段、`_generated_new_text_` 等输出格式。

---

## 二、SID 码本训练（SID_generation）

### 2.1 核心创新

#### 对比学习引入共购信息

- **方案**：采用 CLIP 风格对比学习，以共购商品为正样本对，拉近同购商品的语义表示。
- **实现**：`forward_clip`、`train_clip_i2i` 数据、pair_code_loss 等。
- **效果**：提升语义 ID 的语义一致性与检索相关性。

#### Restart Unused Codes

- **背景**：码本中部分 codeword 长期未被使用，导致利用率偏低。
- **方案**：对 `cluster_size_ema < 1` 的 codeword，以当前 batch 中随机采样（带噪声）的向量替换。
- **实现**：`_tile_with_noise` 扩充向量、`usage` mask 选择性更新。
- **配置**：`restart_unused_codes: True`（默认开启）。

#### Sinkhorn 软分配

- **方案**：训练时对距离做标准化后，使用 Sinkhorn 迭代得到软分配，再取 argmax 得到离散索引；推理时用 argmin。
- **效果**：训练早期提供更平滑的 assignment，有利于稳定收敛。
- **配置**：`use_sinkhorn=True`（训练时默认开启）。

#### EMA 码本更新

- **方案**：`cluster_size_ema` 与 `embed_ema` 做指数移动平均更新，配合 Laplacian smoothing 归一化。
- **效果**：码本更新更平滑，减弱梯度噪声影响。
- **配置**：`VQ_ema: True` 时启用（当前配置为 False）。

#### KMeans 初始化

- **方案**：使用 `residual_kmeans` 对 RQ 各层码本做残差式 KMeans 初始化，替代随机初始化。
- **效果**：提高码本利用率与收敛速度。
- **说明**：默认 `kmeans_init=True`，但 `init_embed_` 调用已注释，当前实际使用 kaiming 初始化。

#### 码本防碰撞（两种模式）

- **背景**：多个商品映射到同一 SID 时产生碰撞，需重新分配以控制每桶商品数。
- **方案**：两阶段流程——阶段一生成候选，阶段二统一防碰撞分配。
- **两种模式**：

| 模式 | 说明 | 实现 |
|------|------|------|
| **KNN** | 基于模型语义相似度，对第 3 级码本做 KNN 排序作为候选 | `generate_candidates_lv3.py --mode knn` |
| **Random** | 不依赖模型，对第 3 级码本随机采样作为候选 | `generate_candidates_lv3.py --mode random` |

- **配置**：`max_per_bucket` 控制每桶最大商品数（默认 5）。

### 2.2 其他优化

- **码本利用率统计**：`compute_codebook_utilization` 监控各层码本使用情况。
- **NPU 支持**：与 algr 一致，支持多 NPU 分布式训练。

---

## 三、创新点一览表

| 模块 | 创新/优化 | 状态 |
|------|-----------|------|
| algr | LM Head 优化 | ✅ 默认开启 |
| algr | 动态 Beam Search | ✅ 默认开启 |
| algr | 训练与推理一体化 | ✅ 支持 |
| algr | NPU 并行 | ✅ 支持 |
| algr | Dataset Streaming | ✅ 支持 |
| algr | dense / scratch / from pretrain | ✅ 支持 |
| algr | pretrain / SFT 任务 | ✅ 支持 |
| SID_generation | 对比学习 + 共购信息 | ✅ 核心 |
| SID_generation | 码本防碰撞（KNN/Random） | ✅ 支持 |
| SID_generation | Restart Unused Codes | ✅ 默认开启 |
| SID_generation | Sinkhorn 软分配 | ✅ 默认开启 |
| SID_generation | EMA 码本更新 | 可配置（当前关闭） |
| SID_generation | KMeans 初始化 | 可配置（当前未调用） |

---

## 四、相关代码位置

| 内容 | 路径 |
|------|------|
| LM Head 优化 | `algr/models/qwen2_5/modeling_qwen.py` 约 1235–1269 行 |
| 动态 Beam Search | `algr/models/qwen2_5/modeling_qwen.py` 约 919–935 行 |
| 分布式与 NPU | `algr/utils/dist_utils.py` |
| 加载模式 | `algr/runner.py` 约 120–144 行 |
| Restart Unused / EMA / Sinkhorn | `SID_generation/rqvae_embed/quantizations.py` |
| 对比学习 | `SID_generation/rqvae_embed/rqvae_clip.py` |
| 码本防碰撞 | `SID_generation/collision_resolve.py`、`generate_candidates_lv3.py` |
| 配置说明 | `algr/docs/CONFIG_GUIDE.md` |
