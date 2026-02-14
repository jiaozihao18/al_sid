# U2I 推荐模型训练指南

本文档介绍 u2i 目录下各个推荐模型的架构特点、训练流程和关键区别。

## 目录

- [BaseModel](#basemodel)
- [SASRec](#sasrec)
- [BERT4Rec](#bert4rec)
- [SASRecAddFeat](#sasrecaddfeat)
- [HSTU Lite](#hstu-lite)
- [模型对比总结](#模型对比总结)

---

## BaseModel

### 架构特点

- **最简单的基线模型**
- 使用平均池化聚合序列信息
- 包含用户表征增强的 MLP

### 核心组件

```python
# basemodel.py
- item_emb: Item Embedding
- position_emb: Position Embedding
- user_mlp: 3层MLP用于用户表征增强
```

### 训练流程

1. **输入表示**
   ```python
   seq_emb = item_emb + pos_emb  # [B, L, D]
   ```

2. **序列聚合**
   ```python
   # 使用masked mean pooling
   user_emb = (seq_emb * mask).sum(dim=1) / valid_length
   ```

3. **用户表征增强**
   ```python
   user_emb = user_mlp(user_emb)  # 通过MLP增强
   ```

4. **损失函数**
   - 交叉熵损失（对比学习）
   - L2正则化

### 代码位置

- 模型定义：`u2i/model/basemodel.py`
- 训练脚本：`u2i/run.py`

---

## SASRec

### 架构特点

- **Self-Attentive Sequential Recommendation**
- 使用 Transformer 的自注意力机制
- **单向（Causal）Attention**：只能看到历史信息
- 适合自回归预测任务

### 核心组件

```python
# sasrec.py
- Multi-head Self-Attention（带Causal Mask）
- PointWiseFeedForward
- Layer Normalization
- Residual Connection
```

### 关键机制

#### 1. Causal Mask（因果掩码）

```python
# 第71行
attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# 上三角矩阵，防止看到未来信息
```

**Mask 矩阵示例（4x4）：**
```
[[False, True,  True,  True ],  位置0只能看自己
 [False, False, True,  True ],  位置1可以看0,1
 [False, False, False, True ],  位置2可以看0,1,2
 [False, False, False, False]]  位置3可以看0,1,2,3
```

#### 2. 标准 Self-Attention

```python
# 计算流程
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
masked_scores = scores.masked_fill(causal_mask, -inf)
weights = softmax(masked_scores)  # 概率分布
output = weights @ V
```

### 训练流程

1. **输入表示**
   ```python
   seq_emb = item_emb + pos_emb  # [B, L, D]
   ```

2. **Transformer Blocks**
   ```python
   for each block:
       norm_x = LayerNorm(seq_emb)
       attn_output = MultiHeadAttention(norm_x, causal_mask)
       seq_emb = seq_emb + attn_output
       
       norm_y = LayerNorm(seq_emb)
       ffn_output = FeedForward(norm_y)
       seq_emb = seq_emb + ffn_output
   ```

3. **用户表示提取**
   ```python
   last_valid_idx = seq_mask.sum(dim=1) - 1
   user_emb = output[torch.arange(batch_size), last_valid_idx, :]
   ```

4. **损失函数**
   - 交叉熵损失（正样本 vs 负样本）
   - L2正则化

### 代码位置

- 模型定义：`u2i/model/sasrec.py`
- 训练脚本：`u2i/run.py`

---

## BERT4Rec

### 架构特点

- **Bidirectional Encoder Representations from Transformers for Recommendation**
- 使用**双向（Bidirectional）Attention**
- 可以看到整个序列的上下文（包括未来位置）
- 适合序列理解任务

### 核心组件

```python
# bert4rec_lite.py
- Multi-head Self-Attention（双向，无Causal Mask）
- PointWiseFeedForward
- Layer Normalization
- Residual Connection
```

### 关键机制

#### 1. 双向 Attention

```python
# 第72行：关键区别！
attn_mask=None,  # None表示双向attention，允许所有位置相互看到
key_padding_mask=key_padding_mask  # 只mask padding位置
```

**Attention 矩阵示例（4x4）：**
```
[[False, False, False, False],  位置0可以看到所有位置（包括未来）
 [False, False, False, False],  位置1可以看到所有位置
 [False, False, False, False],  位置2可以看到所有位置
 [False, False, False, False]]  位置3可以看到所有位置
```

### 训练流程

1. **输入表示**（与 SASRec 相同）
   ```python
   seq_emb = item_emb + pos_emb  # [B, L, D]
   ```

2. **Transformer Blocks**（双向）
   ```python
   for each block:
       norm_x = LayerNorm(seq_emb)
       attn_output = MultiHeadAttention(norm_x, attn_mask=None)  # 双向！
       seq_emb = seq_emb + attn_output
       
       norm_y = LayerNorm(seq_emb)
       ffn_output = FeedForward(norm_y)
       seq_emb = seq_emb + ffn_output
   ```

3. **用户表示提取**（与 SASRec 相同）
   ```python
   last_valid_idx = seq_mask.sum(dim=1) - 1
   user_emb = output[torch.arange(batch_size), last_valid_idx, :]
   ```

4. **损失函数**（与 SASRec 相同）
   - 交叉熵损失
   - L2正则化

### 代码位置

- 模型定义：`u2i/model/bert4rec_lite.py`
- 训练脚本：`u2i/run.py`

---

## SASRecAddFeat

### 架构特点

- **SASRec 的扩展版本**
- 在 SASRec 基础上**添加了额外的特征（Codebook Features）**
- 使用 item 的层级分类信息（codebook_lv1, lv2, lv3）

### 核心组件

```python
# sasrec_addfeat.py
- Item Embedding
- Codebook Embeddings（3个层级）
- Multi-head Self-Attention（带Causal Mask）
- PointWiseFeedForward
```

### 关键机制

#### 1. 特征融合

```python
# 第61-66行
user_seq_emb = self.item_emb(user_seq_tensor[:,:,0])  # Item ID embedding

# 累加3个层级的 codebook embedding
for i in range(1, feat_num):  # feat_num = 4
    codebook_emb_layer = self.codebook_embs[i - 1]
    ids = user_seq_tensor[:, :, i]  # codebook 特征
    user_seq_emb += codebook_emb_layer(ids)  # 累加
```

#### 2. Codebook Embeddings

```python
# 第23-26行
self.codebook_embs = nn.ModuleList([
    nn.Embedding(8192 + 1, hidden_units, padding_idx=0)
    for _ in range(3)  # lv1, lv2, lv3 三个层级
])
```

### 输入数据格式

```python
# user_seq: [B, seq_len, 4]
# 第0列：item_id
# 第1列：codebook_lv1
# 第2列：codebook_lv2
# 第3列：codebook_lv3
```

### 训练流程

1. **输入表示**（带特征）
   ```python
   seq_emb = item_emb(item_id) + codebook_emb1(lv1) + codebook_emb2(lv2) + codebook_emb3(lv3)
   seq_emb = seq_emb + pos_emb
   ```

2. **Transformer Blocks**（与 SASRec 相同）
   - Causal Self-Attention
   - Feed-Forward Network

3. **用户表示提取**（与 SASRec 相同）

4. **损失函数**（与 SASRec 相同）

### 代码位置

- 模型定义：`u2i/model/sasrec_addfeat.py`
- 数据预处理：`u2i/data/process.py`（`process_sequence_with_feat`）

---

## HSTU Lite

### 架构特点

- **Hierarchical Spatial-Temporal Unit（简化版）**
- 使用 **PAA（Pointwise Aggregated Attention）** 替代标准 Self-Attention
- **相对位置编码**（T5风格）
- **门控机制**
- 使用 **SiLU 激活函数**（替代 Softmax）

### 核心组件

```python
# hstu_lite.py
- HSTUBlock（三层结构）
  - Pointwise Projection
  - Spatial Aggregation (PAA)
  - Pointwise Transformation
- RelativeAttentionBias（相对位置编码）
- PointwiseAggregatedAttention（PAA）
```

### 关键机制

#### 1. HSTU Block 三层结构

```python
# 第120-136行
def forward(self, x):
    # 1. Pointwise Projection
    x_proj = silu(f1(x))  # [B, L, D] -> [B, L, 4D]
    u, v, q, k = split(x_proj)  # 分成4个部分
    
    # 2. Spatial Aggregation (PAA)
    av = pointwise_attn(v, k, q)  # PAA聚合
    
    # 3. Pointwise Transformation
    y = f2(norm(av * u))  # 门控+归一化+变换
    
    return y + x  # 残差连接
```

#### 2. PAA（Pointwise Aggregated Attention）

```python
# 第75-99行
def forward(self, v, k, q, mask=None):
    # 1. 计算注意力分数
    attention_scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
    
    # 2. 添加相对位置偏置
    rab = relative_attention_bias(L, L)
    att_w_bias = attention_scores + rab
    
    # 3. SiLU 激活（关键区别！）
    weights = silu(att_w_bias)  # 不是 softmax！
    
    # 4. 聚合
    av = weights @ v
    return av
```

**与标准 Attention 的区别：**
- 标准：`softmax(scores) @ v`（概率分布）
- PAA：`silu(scores + relative_bias) @ v`（非概率分布）

#### 3. 相对位置编码

```python
# 第10-54行
class RelativeAttentionBias:
    def forward(self, query_length, key_length):
        relative_position = memory_pos - context_pos
        bucket = _relative_position_bucket(relative_position)
        bias = self.relative_attention_bias(bucket)
        return bias
```

#### 4. 门控机制

```python
# 第133行
y = f2(norm(av * u))  # u 作为门控信号
```

### 训练流程

1. **输入表示**（与 SASRec 相同）
   ```python
   seq_emb = item_emb + pos_emb  # [B, L, D]
   ```

2. **HSTU Blocks**
   ```python
   for each HSTUBlock:
       # Pointwise Projection
       x_proj = silu(f1(x))
       u, v, q, k = split(x_proj)
       
       # Spatial Aggregation (PAA)
       av = PAA(v, k, q, mask)
       
       # Pointwise Transformation
       y = f2(norm(av * u))
       
       x = x + y  # 残差连接
   ```

3. **用户表示提取**（与 SASRec 相同）

4. **损失函数**（与 SASRec 相同）

### 缺失的特性（Lite版本）

- ❌ **相对时间编码**（Relative Time Attention）
- ❌ **时间戳输入处理**

原始 HSTU 应该包含时间戳信息，当前实现只有相对位置编码。

### 代码位置

- 模型定义：`u2i/model/hstu_lite.py`
- 训练脚本：`u2i/run.py`

---

## 模型对比总结

### 架构对比

| 模型 | Attention 类型 | 位置编码 | 激活函数 | 特殊机制 |
|------|---------------|---------|---------|---------|
| **BaseModel** | 无 | 绝对位置 | - | Masked Mean Pooling |
| **SASRec** | 单向（Causal） | 绝对位置 | Softmax | Causal Mask |
| **BERT4Rec** | 双向 | 绝对位置 | Softmax | 无 Mask（双向） |
| **SASRecAddFeat** | 单向（Causal） | 绝对位置 | Softmax | Codebook Features |
| **HSTU Lite** | 单向（Causal） | 相对位置 | SiLU | PAA + 门控 |

### Attention 机制对比

| 模型 | Attention 公式 | 归一化方式 |
|------|---------------|-----------|
| **SASRec** | `softmax(QK^T / √d) @ V` | Softmax（概率分布） |
| **BERT4Rec** | `softmax(QK^T / √d) @ V` | Softmax（概率分布） |
| **HSTU Lite** | `silu(QK^T / √d + R) @ V` | SiLU（非概率分布） |

### 训练流程对比

所有模型的训练流程基本相同：

1. **数据准备**
   ```python
   - user_history: 用户历史序列
   - target_item: 正样本
   - negative_items: 负样本（随机采样）
   ```

2. **前向传播**
   ```python
   user_emb = model.get_user_emb(user_history)
   pos_emb = model.get_item_emb(target_item)
   neg_emb = model.get_item_emb(negative_items)
   ```

3. **损失计算**
   ```python
   pos_score = (user_emb * pos_emb).sum(dim=1)
   neg_score = (user_emb.unsqueeze(1) * neg_emb.unsqueeze(0)).sum(dim=-1)
   logits = torch.cat([pos_score, neg_score], dim=1)
   ce_loss = F.cross_entropy(logits, labels)
   loss = ce_loss + L2_regularization
   ```

4. **反向传播**
   ```python
   loss.backward()
   optimizer.step()
   ```

### 适用场景

| 模型 | 适用场景 | 优势 |
|------|---------|------|
| **BaseModel** | 基线对比 | 简单快速 |
| **SASRec** | 自回归预测 | 单向，适合预测下一个item |
| **BERT4Rec** | 序列理解 | 双向，更好的上下文理解 |
| **SASRecAddFeat** | 有特征数据 | 利用item的层级分类信息 |
| **HSTU Lite** | 序列推荐 | PAA机制，相对位置编码 |

### 数据格式要求

| 模型 | 输入格式 | 数据文件 |
|------|---------|---------|
| **BaseModel** | `[B, L, C]` | `s1_tiny_with_feat.pt` |
| **SASRec** | `[B, L, C]` | `s1_tiny_with_feat.pt` |
| **BERT4Rec** | `[B, L, C]` | `s1_tiny_with_feat.pt` |
| **SASRecAddFeat** | `[B, L, 4]` | `s1_tiny_with_feat.pt` |
| **HSTU Lite** | `[B, L, C]` | `s1_tiny_with_feat.pt` |

---

## 训练命令示例

### SASRec
```bash
python3 run.py --model_name=sasrec \
    --lr=5e-3 \
    --num_heads=4 \
    --num_blocks=8 \
    --epochs=5 \
    --data_path=/path/to/s1_tiny_with_feat.pt
```

### BERT4Rec
```bash
python3 run.py --model_name=bert4rec \
    --lr=5e-3 \
    --num_heads=4 \
    --num_blocks=8 \
    --epochs=5 \
    --data_path=/path/to/s1_tiny_with_feat.pt
```

### SASRecAddFeat
```bash
python3 run.py --model_name=sasrec_addfeat \
    --lr=5e-3 \
    --num_heads=4 \
    --num_blocks=8 \
    --epochs=5 \
    --data_path=/path/to/s1_tiny_with_feat.pt
```

### HSTU Lite
```bash
python3 run.py --model_name=hstu \
    --lr=5e-3 \
    --num_heads=4 \
    --num_blocks=8 \
    --epochs=5 \
    --data_path=/path/to/s1_tiny_with_feat.pt
```

---

## 关键参数说明

### 通用参数

- `--item_count`: Item总数（必须等于max_item_id）
- `--maxlen`: 序列最大长度（默认100）
- `--hidden_units`: 隐藏层维度（默认64）
- `--num_heads`: Attention头数（默认1）
- `--num_blocks`: Transformer层数（默认1）
- `--batch_size`: 批次大小（默认512）
- `--num_neg_samples`: 负样本数量（默认20）
- `--lr`: 学习率（默认5e-3）
- `--dropout_rate`: Dropout率（默认0.05）

### 损失函数权重

- `user_l2_lambda`: 用户embedding的L2正则权重（默认0.05）
- `item_l2_lambda`: Item embedding的L2正则权重（默认0.5）

---

## 注意事项

1. **负采样问题**：当前实现可能采样到target_item本身或历史中已出现的item，建议排除这些item。

2. **序列顺序**：`process.py`中的`[::-1]`反序操作需要确认数据的时间顺序约定。

3. **item_num设置**：`item_num`必须等于数据中的最大item ID，不能大于也不能小于。

4. **数据格式**：
   - 标准模型：`user_history`和`target_item`都是逗号分隔的整数
   - SASRecAddFeat：`user_history`格式为`item_id|feat1|feat2|feat3`

---

## 参考资料

- SASRec: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- BERT4Rec: [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations](https://arxiv.org/abs/1904.06690)
- HSTU: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2512.16487)

---

*最后更新：2025-02-13*
