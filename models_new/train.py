import argparse
import json
import os
from typing import Any, Dict

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

import dist_utils
from data import LDPODataProcess
from ldpo_data_collator import LDPODataCollator
from trainer import RecTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ----------------- 1. 训练参数与自定义参数 -----------------
    parser_ta = HfArgumentParser(TrainingArguments)
    (training_args,) = parser_ta.parse_dict(cfg["training_args"], allow_extra_keys=False)

    # ----------------- 2. 初始化分布式 / 设备（NPU） -----------------
    dist_utils.init_distributed_mode(npu_id=int(cfg.get("npu_id", 0)))

    # ----------------- 3. tokenizer & model -----------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], padding_side="left")
    # 扩展词表：[SEP] + C0~C65535
    print("Old tokenizer length: ", len(tokenizer))
    special_tokens_dict = {"additional_special_tokens": tokenizer.all_special_tokens + ["[SEP]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_tokens(["C%d" % i for i in range(0, 2 * 32768)])
    print("New tokenizer length: ", len(tokenizer))

    local_rank = int(os.environ.get("LOCAL_RANK", int(cfg.get("npu_id", 0))))
    device_map = f"npu:{max(local_rank, 0)}"
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], device_map=device_map)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

    # ----------------- 4. 构建数据集 -----------------
    data_file = cfg["data_file"]
    ldpo_data_file = cfg.get("ldpo_data_file")
    streaming = False
    num_proc = int(cfg.get("num_proc", 1))

    def load_and_tokenize(path: str, preprocess: LDPODataProcess) -> Any:
        print(f"loading dataset from {path}...")
        if not os.path.isfile(path):
            raise ValueError(f"data_file '{path}' 必须是本地 csv 文件路径")
        dataset = load_dataset("csv", data_files=path, split="train", streaming=streaming)
        print("processing dataset...")
        return dataset.map(
            preprocess,
            batched=False,
            num_proc=num_proc,
            remove_columns=[
                cfg.get("instruction_column", "system"),
                cfg.get("input_column", "user"),
                cfg.get("output_column", "answer"),
            ],
        )

    preprocess_common_args = {
        "instruction_column": cfg.get("instruction_column", "system"),
        "input_column": cfg.get("input_column", "user"),
        "output_column": cfg.get("output_column", "answer"),
        "max_length": cfg.get("max_length", 1068),
        "max_source_length": cfg.get("max_source_length", 1024),
        "max_target_length": cfg.get("max_target_length", 32),
        "ldpo_m": cfg.get("ldpo_m", 3),
    }

    preprocess_ce = LDPODataProcess(
        {
            **preprocess_common_args,
            "training_mode": None,
        },
        tokenizer,
        is_train=True,
    )
    tokenized_ce = load_and_tokenize(data_file, preprocess_ce)

    train_dataset = tokenized_ce

    # 统一字段（sample_type/ldpo_*）需要保留，避免被 Trainer 自动裁剪
    training_args.remove_unused_columns = False

    if ldpo_data_file:

        preprocess_ldpo = LDPODataProcess(
            {
                **preprocess_common_args,
                "training_mode": "iap_ldpo",
            },
            tokenizer,
            is_train=True,
        )
        tokenized_ldpo = load_and_tokenize(ldpo_data_file, preprocess_ldpo)
        train_dataset = concatenate_datasets([tokenized_ce, tokenized_ldpo])

    # 统一用 4D item-aware collator；CE 样本会退化为普通因果可见性
    collator = LDPODataCollator(
        tokenizer=tokenizer,
        padding=True,
    )

    # ----------------- 5. collator & Trainer -----------------
    trainer = RecTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        ldpo_alpha=float(cfg.get("ldpo_alpha", 1.0)),
        ldpo_beta=float(cfg.get("ldpo_beta", 1.0)),
        ldpo_only=bool(cfg.get("ldpo_only", False)),
    )

    # ----------------- 6. 开始训练 -----------------
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()

