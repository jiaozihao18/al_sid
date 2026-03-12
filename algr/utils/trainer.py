from transformers import Seq2SeqTrainer, GenerationConfig
from torch import nn
from typing import Optional, Union, Dict, Any, List, Tuple
import torch
from .log import logger
import copy
import numpy as np
from utils.predict import PredictWriter, create_predict_writer
from torch.utils.data import DataLoader

from utils.mixed_batch_dataset import AlternatingBatchIterable
import inspect

class GRSTrainer(Seq2SeqTrainer):

    predict_output: Optional[Dict[str, Any]] = None
    predict_writer: Optional[PredictWriter] = None
    ldpo_train_dataset: Optional[Any] = None
    ldpo_data_collator: Optional[Any] = None
    ldpo_ratio: float = 0.5
    ldpo_seed: int = 42
    ldpo_start_with: str = "ce"

    def __init__(self,
                 predict_output: Optional[Dict[str, Any]] = None,
                 ldpo_train_dataset: Optional[Any] = None,
                 ldpo_data_collator: Optional[Any] = None,
                 ldpo_ratio: float = 0.5,
                 ldpo_seed: int = 42,
                 ldpo_start_with: str = "ce",
                 **kwargs):

        model = kwargs.get('model')
        args = kwargs.get('args')
        if args.generation_config is not None and isinstance(args.generation_config, dict):
            if hasattr(model, "generation_config") and model.generation_config is not None:
                generation_config: GenerationConfig = copy.deepcopy(model.generation_config)
                generation_config.update(**args.generation_config)
                args.generation_config = generation_config
                logger.info("merge model default and user defined generation_config: %s", str(generation_config))
            else:
                args.generation_config = GenerationConfig.from_dict(args.generation_config)

        super(GRSTrainer, self).__init__(**kwargs)
        self.predict_output = None
        self.predict_writer = None
        self.ldpo_train_dataset = ldpo_train_dataset
        self.ldpo_data_collator = ldpo_data_collator
        self.ldpo_ratio = ldpo_ratio
        self.ldpo_seed = ldpo_seed
        self.ldpo_start_with = ldpo_start_with

        if self.args.do_predict:
            self.predict_output = predict_output
            self.predict_writer = create_predict_writer(self.predict_output)
            if self.predict_output is None:
                raise ValueError("predict_output isn't set")
            if not self.args.predict_with_generate or self.args.prediction_loss_only:
                if not hasattr(self.model, "predict_trace_dict"):
                    raise ValueError(f"method {self.model.__class__.__name__}.predict_trace_dict() isn't defined, "
                                     f"may you should set {'predict_with_generate=True' if not self.args.predict_with_generate else 'prediction_loss_only=False'}")
            if not self.args.remove_unused_columns:
                self.accelerator.device_placement = False

    def get_train_dataloader(self) -> DataLoader:
        """
        支持交替训练：
        - 默认：与 HF Seq2SeqTrainer 相同
        - 若传入 ldpo_train_dataset + ldpo_data_collator：构建 CE/LDPO 两个内部 DataLoader，
          然后用 Iterable 交替产出 batch。
        """
        if self.ldpo_train_dataset is None or self.ldpo_data_collator is None:
            return super().get_train_dataloader()

        # 1) CE dataloader：沿用 Trainer 的默认逻辑（含 sampler / drop_last 等）
        ce_loader = super().get_train_dataloader()

        # 2) LDPO dataloader：复用 Trainer 的 sampler 逻辑（DistributedSampler 等）
        #    这里不能复用 super().get_train_dataloader()，因为那会绑到 self.train_dataset；
        #    因此手动构建，尽量对齐 Trainer 的常见参数。
        # transformers 不同版本 _get_train_sampler 的签名不一致，这里做兼容：
        # - 新版：_get_train_sampler(train_dataset)
        # - 旧版：_get_train_sampler()
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
            batch_size=self._train_batch_size,
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

        # 3) 用 batch_size=None 的 DataLoader 包一层，让 Trainer 以为这是“一个 dataloader”
        mixed_loader = DataLoader(mixed_iterable, batch_size=None)
        return mixed_loader

    ## Generation Process:
    # 1. Define a writer based on the predict_output setting
    # 2. Write the output to the output.txt file
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        ignored_inputs = {}
        if not self.args.remove_unused_columns:
            self._set_signature_columns_if_needed()
            signature_columns = self._signature_columns
            dataset_column_names = list(inputs.keys())

            ignored_columns = list(set(dataset_column_names) - set(signature_columns))

            ignored_inputs = {ignored_column: inputs.pop(ignored_column) for ignored_column in ignored_columns}
            # inputs = send_to_device(inputs,
            #                         device=self.accelerator.device,
            #                         non_blocking=self.accelerator.non_blocking)

        results = super(GRSTrainer, self).prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        if not self.args.do_predict:
            return results

        trace_dict = None
        if hasattr(self.model, "predict_trace_dict"):
            trace_dict = self.model.predict_trace_dict()

        if trace_dict is None:
            trace_dict = {}
    

        if not trace_dict and self.args.predict_with_generate and not prediction_loss_only:
            output_columns = self.predict_output.get("columns", [])
            loss, generated_tokens, labels = results

            batch_size = self.args.per_device_eval_batch_size
            if "input_ids" in inputs:
                batch_size = inputs["input_ids"].shape[0]
            else:
                for key, value in inputs.items():
                    batch_size = value.shape[0]

            if "_generated_tokens_" in output_columns:
                if batch_size != generated_tokens.shape[0]:
                    trace_dict["_generated_tokens_"] = generated_tokens.reshape(
                        [batch_size, -1, generated_tokens.shape[-1]])
                else:
                    trace_dict["_generated_tokens_"] = generated_tokens

            # 去掉input_ids
            if "_generated_new_tokens_" in output_columns:
                generated_new_tokens = generated_tokens[:, inputs["input_ids"].size(-1):]
                if batch_size != generated_new_tokens.shape[0]:
                    trace_dict["_generated_new_tokens_"] = generated_new_tokens.reshape(
                        [batch_size, -1, generated_new_tokens.shape[-1]])
                else:
                    trace_dict["_generated_new_tokens_"] = generated_new_tokens

            if not output_columns or "_generated_text_" in output_columns:
                generated_text = self.processing_class.batch_decode(generated_tokens, skip_special_tokens=self.predict_output.get("skip_special_tokens", True))
                if batch_size != len(generated_text):
                    generated_text = np.array(generated_text).reshape([batch_size, -1]).tolist()
                trace_dict["_generated_text_"] = generated_text

            # 去掉input_ids
            if not output_columns or "_generated_new_text_" in output_columns:
                generated_new_tokens = generated_tokens[:, inputs["input_ids"].size(-1): ]
                generated_new_text = self.processing_class.batch_decode(generated_new_tokens, skip_special_tokens=self.predict_output.get("skip_special_tokens", True))
                if batch_size != len(generated_new_text):
                    generated_new_text = np.array(generated_new_text).reshape([batch_size, -1]).tolist()
                trace_dict["_generated_new_text_"] = generated_new_text

            if not output_columns:
                trace_dict.update(ignored_inputs)
                trace_dict.update(inputs)
            else:
                for output_column in output_columns:
                    if output_column in ignored_inputs:
                        trace_dict[output_column] = ignored_inputs[output_column]
                    if output_column in inputs:
                        trace_dict[output_column] = inputs[output_column]
        self.predict_writer.write(trace_dict)
        if self.compute_metrics is not None:
            return results
        # 避免gather和占用显存
        return None, None, None
