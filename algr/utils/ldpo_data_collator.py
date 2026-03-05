from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers.trainer_pt_utils import LabelSmoother
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


IGNORE_INDEX = LabelSmoother.ignore_index


@dataclass
class LDPODataCollator(DataCollatorMixin):
    """
    DataCollator 专用于 IAP + LDPO 训练。

    约定每个样本至少包含:
    - input_ids: List[int]，长度 L
    - labels:   List[int]，长度 L
    - ldpo_item_index: List[int]，长度 L，历史为 -1，item 为 [0..N-1]

    可选:
    - ldpo_item_groups: List[int]，长度 N，按 item 维度的分组 (0 为 padding，其它 1..J)
    - ldpo_num_items:   int，每条样本真实 item 数
    - ldpo_m:           int，每个 item 的 codeword 数 (可变时也可缺省)
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True

    def __call__(self, features: List[Dict[str, Any]], return_tensors: Optional[str] = None) -> Dict[str, Any]:
        # 1. 计算 batch 内的最大长度，用于 padding
        max_length = max(len(f["input_ids"]) for f in features)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        batch_input_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_item_index: List[List[int]] = []
        batch_item_groups: List[Optional[List[int]]] = []
        batch_num_items: List[Optional[int]] = []
        batch_m: List[Optional[int]] = []

        for f in features:
            input_ids: List[int] = f["input_ids"]
            labels: List[int] = f.get("labels", [IGNORE_INDEX] * len(input_ids))
            item_index: List[int] = f.get("ldpo_item_index", [-1] * len(input_ids))

            if not (len(input_ids) == len(labels) == len(item_index)):
                raise ValueError(
                    f"Length mismatch in LDPODataCollator: "
                    f"len(input_ids)={len(input_ids)}, len(labels)={len(labels)}, len(ldpo_item_index)={len(item_index)}"
                )

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [pad_token_id] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len
                item_index = item_index + [-1] * pad_len

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_item_index.append(item_index)

            batch_item_groups.append(f.get("ldpo_item_groups"))
            batch_num_items.append(f.get("ldpo_num_items"))
            batch_m.append(f.get("ldpo_m"))

        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
        item_index_tensor = torch.tensor(batch_item_index, dtype=torch.long)

        batch: Dict[str, Any] = {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "ldpo_item_index": item_index_tensor,
        }

        # 2. collate 按 item 维度的字段
        valid_groups = [g for g in batch_item_groups if g is not None]
        if valid_groups:
            max_items = max(len(g) for g in valid_groups)
            groups_tensor = torch.zeros((len(features), max_items), dtype=torch.long)
            for i, g in enumerate(batch_item_groups):
                if g is None:
                    continue
                g_tensor = torch.tensor(g, dtype=torch.long)
                groups_tensor[i, : g_tensor.size(0)] = g_tensor
            batch["ldpo_item_groups"] = groups_tensor

        valid_num_items = [n for n in batch_num_items if n is not None]
        if valid_num_items:
            batch["ldpo_num_items"] = torch.tensor(
                [n if n is not None else 0 for n in batch_num_items], dtype=torch.long
            )

        valid_m = [m for m in batch_m if m is not None]
        if valid_m:
            batch["ldpo_m"] = torch.tensor(
                [m if m is not None else 0 for m in batch_m], dtype=torch.long
            )

        # 3. 构造 4D item-aware attention mask: [B, 1, L, L]
        batch_size, seq_len = item_index_tensor.shape
        dtype = torch.float32
        minv = torch.finfo(dtype).min
        attention_mask = torch.full((batch_size, 1, seq_len, seq_len), fill_value=minv, dtype=dtype)

        for b in range(batch_size):
            indices = item_index_tensor[b]  # [L]
            is_history = indices.eq(-1)  # [L]

            # key 是否历史
            key_is_history = is_history.unsqueeze(0).expand(seq_len, seq_len)  # [L, L]
            query_item = indices.unsqueeze(1)  # [L, 1]
            key_item = indices.unsqueeze(0)  # [1, L]

            same_item = query_item.eq(key_item) & query_item.ne(-1)  # [L, L]，同一 item 且非历史

            # 因果约束 j <= i
            causal = torch.ones((seq_len, seq_len), dtype=torch.bool)
            causal = torch.tril(causal, diagonal=0)

            # 允许关注：因果 && (key 是历史 || 同 item)
            allowed = causal & (key_is_history | same_item)  # [L, L]

            mask_2d = attention_mask[b, 0]
            mask_2d[allowed] = 0.0
            # 其它位置保持为 minv

        batch["attention_mask"] = attention_mask

        return batch

