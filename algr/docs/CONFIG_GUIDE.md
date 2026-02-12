# Algr 训练配置指南

本文档汇总 algr 的训练模式、模型加载方式、数据模式及配置参数说明。

---

## 1. 任务模式（`training_args`）

| 参数 | 含义 | 配置位置 |
|------|------|----------|
| `do_train` | 训练 | `training_args.do_train` |
| `do_eval` | 评估 | `training_args.do_eval` |
| `do_predict` | 推理 | `training_args.do_predict` |

### 互斥规则

- 必须**恰好一个**为 `True`，**或**
- `do_train` 和 `do_eval` **同时**为 `True`（训练 + 评估）

**示例**：`"do_train": true` 或 `"do_predict": true`

---

## 2. 模型加载方式（`custom_args.load_func`）

| 值 | 含义 | 配置位置 |
|------|------|----------|
| `"scratch"` | 从 config 初始化空模型，随机权重 | `custom_args.load_func` |
| `"dense"` | 加载预训练，随机初始化 embedding，freeze 其他层，只训练 embedding | `custom_args.load_func` |
| 未设置 / 其他 | 全量微调：`from_pretrained` 加载全部权重 | 不配置或非上述值 |

**示例**：`"custom_args": { "load_func": "dense", ... }`

---

## 3. 训练数据模式（`custom_args.training_mode`）

| 值 | 含义 | 配置位置 |
|------|------|----------|
| `"pretrain"` | 预训练：input 直接作为文本，无 instruction 格式 | `custom_args.training_mode` |
| 未设置 / 其他 | 微调：instruction 对话格式（system/user/answer） | 不配置或非 `"pretrain"` |

**示例**：`"custom_args": { "training_mode": "pretrain", ... }`

---

## 4. 互斥与组合关系

| 维度 | 互斥关系 |
|------|----------|
| 任务 | `do_train` / `do_eval` / `do_predict` 须满足上述规则 |
| 加载 | `load_func` 三选一：`scratch` / `dense` / 默认全量 |
| 数据 | `training_mode` 二选一：`pretrain` / 默认 train |

### 常见组合

| 场景 | 配置 |
|------|------|
| 预训练 | `training_mode: "pretrain"` + `load_func: "scratch"` 或默认 |
| 微调（只训 embedding） | `load_func: "dense"` |
| 全量微调 | `load_func` 不配置或非 scratch/dense |
| 推理 | `do_predict: true` |

---

## 5. 其他相关配置

| 参数 | 说明 |
|------|------|
| `model_type` | `"qwen2_5"` 或 `"t5"` |
| `streaming` | 是否流式加载数据 |
| `instruction_column` / `input_column` / `output_column` | 数据列名（非 pretrain 时） |
| `predict_output` | 推理输出配置（`do_predict` 时） |

---

## 6. 配置示例

### 预训练

```json
{
  "custom_args": {
    "training_mode": "pretrain",
    "load_func": "scratch",
    "instruction_column": "system",
    "input_column": "user",
    "output_column": "answer",
    "max_length": 1068,
    "max_source_length": 1024,
    "max_target_length": 32
  },
  "training_args": {
    "do_train": true,
    ...
  }
}
```

### 只训练 Embedding 层

```json
{
  "custom_args": {
    "load_func": "dense",
    "instruction_column": "system",
    "input_column": "user",
    "output_column": "answer",
    ...
  },
  "training_args": {
    "do_train": true,
    ...
  }
}
```

### 全量微调

```json
{
  "custom_args": {
    "instruction_column": "system",
    "input_column": "user",
    "output_column": "answer",
    ...
  },
  "training_args": {
    "do_train": true,
    ...
  }
}
```

### 推理

```json
{
  "training_args": {
    "do_predict": true,
    ...
  },
  "predict_output": {
    "type": "local",
    "mode": "overwrite",
    "columns": ["answer", "_generated_new_text_", "user", "input_ids"],
    "skip_special_tokens": true
  }
}
```

---

## 7. 常用 training_args 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `max_steps` | 最大训练步数，`-1` 表示不限制 | `12500` / `-1` |
| `save_steps` | 每隔多少步保存 checkpoint，`-1` 表示不按步保存 | `10000` / `-1` |
| `gradient_accumulation_steps` | 梯度累积步数 | `4`（默认 1） |
| `report_to` | 日志上报目标，如 TensorBoard | `"tensorboard"` |
| `dataloader_num_workers` | DataLoader 进程数，streaming 时建议 1 | `1` |
| `ddp_find_unused_parameters` | DDP 是否查找未使用参数 | `false` |

---

## 8. 数据加载说明（streaming）

当 `streaming: true` 时：

- 数据**流式加载**，不会一次性全部读入内存
- `filter` 和 `map`（含 tokenization）为**惰性**，按 batch 按需处理
- 建议将 `dataloader_num_workers` 设为 `1`（受 `num_shards` 限制）
