import argparse
import json
import os
from typing import Any, Dict, Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

import dist_utils
from data import QwenDataProcess
from ldpo_data_collator import LDPODataCollator
from mixed_batch_dataset import AlternatingBatchIterable
from torch.utils.data import DataLoader
import inspect


class RecTrainer(Trainer):
    def __init__(
        self,
        ldpo_train_dataset=None,
        ldpo_data_collator=None,
        ldpo_ratio: float = 0.5,
        ldpo_seed: int = 42,
        ldpo_start_with: str = "ce",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ldpo_train_dataset = ldpo_train_dataset
        self.ldpo_data_collator = ldpo_data_collator
        self.ldpo_ratio = ldpo_ratio
        self.ldpo_seed = ldpo_seed
        self.ldpo_start_with = ldpo_start_with

    def get_train_dataloader(self) -> DataLoader:
        if self.ldpo_train_dataset is None or self.ldpo_data_collator is None:
            return super().get_train_dataloader()

        ce_loader = super().get_train_dataloader()

        try:
            sig = inspect.signature(self._get_train_sampler)
            if len(sig.parameters) >= 1:
                ldpo_sampler = self._get_train_sampler(self.ldpo_train_dataset)
            else:
                ldpo_sampler = self._get_train_sampler()
        except Exception:
            ldpo_sampler = None

        ldpo_loader = DataLoader(
            self.ldpo_train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=ldpo_sampler,
            collate_fn=self.ldpo_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

        mixed_iterable = AlternatingBatchIterable(
            ce_iterable=ce_loader,
            ldpo_iterable=ldpo_loader,
            ldpo_ratio=float(self.ldpo_ratio),
            seed=int(self.ldpo_seed),
            start_with=str(self.ldpo_start_with),
        )

        mixed_loader = DataLoader(mixed_iterable, batch_size=None)
        return mixed_loader


def build_training_args(raw_training_args: Dict[str, Any]) -> TrainingArguments:
    """从 dict 构建 TrainingArguments。"""
    parser = HfArgumentParser(TrainingArguments)
    raw_training_args["output_dir"] = os.path.expanduser(raw_training_args["output_dir"])
    (training_args,) = parser.parse_dict(raw_training_args, allow_extra_keys=False)
    return training_args


def build_preprocess(
    custom_args: Dict[str, Any],
    tokenizer,
    is_train: bool,
    training_mode: Optional[str] = None,
) -> QwenDataProcess:
    """构建 QwenDataProcess；可临时覆盖 training_mode（None / "iap_ldpo"）。"""
    merged = dict(custom_args)
    merged["training_mode"] = training_mode
    return QwenDataProcess(merged, tokenizer, is_train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ----------------- 1. 训练参数与自定义参数 -----------------
    training_args = build_training_args(cfg["training_args"])
    # 只做 train，不做 eval / predict
    training_args.do_train = True
    training_args.do_eval = False
    training_args.do_predict = False

    custom_args: Dict[str, Any] = {
        "instruction_column": cfg.get("instruction_column", "system"),
        "input_column": cfg.get("input_column", "user"),
        "output_column": cfg.get("output_column", "answer"),
        "max_length": cfg.get("max_length", 1068),
        "max_source_length": cfg.get("max_source_length", 1024),
        "max_target_length": cfg.get("max_target_length", 32),
        "ldpo_m": cfg.get("ldpo_m", 3),
        "ldpo_ratio": cfg.get("ldpo_ratio", 0.5),
        "ldpo_only": cfg.get("ldpo_only", False),
    }

    model_name_or_path = cfg["model_name_or_path"]

    # ----------------- 2. 初始化分布式 / 设备（NPU） -----------------
    dist_utils.init_distributed_mode(device_type="npu", args=argparse.Namespace(local_rank=-1))

    # ----------------- 3. tokenizer & model -----------------
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    # 扩展词表：[SEP] + C0~C65535
    print("Old tokenizer length: ", len(tokenizer))
    special_tokens_dict = {"additional_special_tokens": tokenizer.all_special_tokens + ["[SEP]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_tokens(["C%d" % i for i in range(0, 2 * 32768)])
    print("New tokenizer length: ", len(tokenizer))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = f"npu:{max(local_rank, 0)}"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

    # ----------------- 4. 构建数据集 -----------------
    data_file = cfg["data_file"]
    ldpo_data_file = cfg.get("ldpo_data_file")
    streaming = cfg.get("streaming", False)

    def load_and_tokenize(path: str, preprocess: QwenDataProcess) -> Any:
        print(f"📊 loading dataset from {path}...")
        if not os.path.isfile(path):
            raise ValueError(f"data_file '{path}' 必须是本地 csv 文件路径")
        dataset = load_dataset("csv", data_files=path, split="train", streaming=streaming)
        print("🔄 processing dataset...")
        dataset = dataset.filter(preprocess.filter_fn)
        return dataset.map(
            preprocess,
            batched=False,
            remove_columns=[
                custom_args["instruction_column"],
                custom_args["input_column"],
                custom_args["output_column"],
            ],
        )

    preprocess_ce = build_preprocess(custom_args, tokenizer, is_train=True, training_mode=None)
    tokenized_ce = load_and_tokenize(data_file, preprocess_ce)

    ldpo_train_dataset = None
    ldpo_collator = None

    if ldpo_data_file:
        training_args.remove_unused_columns = False

        preprocess_ldpo = build_preprocess(custom_args, tokenizer, is_train=True, training_mode="iap_ldpo")
        ldpo_train_dataset = load_and_tokenize(ldpo_data_file, preprocess_ldpo)
        ldpo_collator = LDPODataCollator(
            tokenizer=tokenizer,
            padding=True,
            ldpo_only=bool(custom_args["ldpo_only"]),
        )

    # ----------------- 5. collator & Trainer -----------------
    ce_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = RecTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ce,
        data_collator=ce_collator,
        tokenizer=tokenizer,
        ldpo_train_dataset=ldpo_train_dataset,
        ldpo_data_collator=ldpo_collator,
        ldpo_ratio=float(custom_args["ldpo_ratio"]),
        ldpo_seed=42,
        ldpo_start_with="ce",
    )

    # ----------------- 6. 开始训练 -----------------
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()

