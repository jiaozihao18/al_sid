#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AUTHOR  :selous
DATE    :2025.9.2
FUNC    :All-in-one script: Extend the tokenizer of Qwen2.5-0.5B (add C0-C65536) and perform full fine-tuning
"""
import os
import argparse
import traceback
from typing import Union
import torch.distributed as dist
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_dataset
from utils.common import EasyDict
from utils.util import convert_args_value_type
from utils.log import logger
from utils.trainer import GRSTrainer
from utils.data_collator import DataCollatorWrapper
from utils import dist_utils

## Define different Runner configurations based on parameters
class Runner:
    config: EasyDict = None
    def __init__(self, config: EasyDict):

        self.config = config
        if self.config.envs:
            for key, value in self.config.envs.items():
                os.environ[str(key)] = str(value)
        # -------------------------------
        # 1. Parsing configuration parameters including custom_args and training_args, which need to be combined with huggingface configuration
        # -------------------------------
        self.training_args, self.custom_args = self.init_args(self.config)
        self.predict_output = self.config.predict_output
        self.gen_kwargs = None

        self.train_dataset = None
        self.ldpo_train_dataset = None
        self.test_dataset = None
        self.preprocess_function = None
        self.ldpo_preprocess_function = None

        self.trainer = None
        self.is_train = (self.training_args.do_train is True)

    def training_args_class(self):
        return Seq2SeqTrainingArguments
    
    def trainer_class(self):
        return GRSTrainer
    
    def create_compute_loss_func(self):
        return None

    def init_args(self, config: EasyDict):
        training_args_class = self.training_args_class()
        if not issubclass(training_args_class, TrainingArguments):
            raise ValueError(f"invalid training args class, it should be inherited from TrainingArguments, "
                            f"current is {training_args_class.__name__}")

        parser = HfArgumentParser(training_args_class)
        training_args = convert_args_value_type(config.get("training_args", {}), training_args_class)
        ## Need to build a usable file name based on the configuration
        training_args["output_dir"] = os.path.expanduser(training_args["output_dir"])

        ## Todo: Create subfolders based on parameters
        training_args["report_to"] = "none"
        training_args, = parser.parse_dict(training_args, False) #Convert training_args into HF format Arguments

        job_types = [training_args.do_train,training_args.do_eval,training_args.do_predict]
        if sum(job_types) != 1 and not (job_types[0] and job_types[1]):
            # train and eval can be set at the same time
            raise ValueError(
                f"one and only one of [do_train, do_eval, do_predict] should be set as True, current is {job_types}")

        custom_args = EasyDict(config.get("custom_args", {}))
        return training_args, custom_args

    def create_preprocess(self):
        if self.config.model_type == 'qwen2_5':
            from models.qwen2_5.data import QwenDataProcess as DataProcess
        elif self.config.model_type == 't5':
            from models.t5.data import T5DataProcess as DataProcess
        preprocess_function = DataProcess(self.custom_args, self.tokenizer, self.is_train)
        return preprocess_function

    def _create_preprocess_with_mode(self, training_mode: str):
        if self.config.model_type == 'qwen2_5':
            from models.qwen2_5.data import QwenDataProcess as DataProcess
        elif self.config.model_type == 't5':
            from models.t5.data import T5DataProcess as DataProcess
        # clone custom_args but override training_mode
        cloned = EasyDict(dict(self.custom_args))
        cloned.training_mode = training_mode
        return DataProcess(cloned, self.tokenizer, self.is_train)
        
    def create_model(self):
        ## Create model according to self.config
        checkpoint_path = self.config.load_checkpoint_from
        device_type = getattr(self.config, 'device_type', None) or 'cuda'
        local_rank = int(os.environ.get('LOCAL_RANK', getattr(self, '_local_rank', 0)))

        if device_type.lower() == 'cpu':
            device_map = "cpu"
        else:
            device_map = dist_utils.get_device_for_model_loading(device_type, local_rank)
        
        if self.config.model_type == 'qwen2_5':
            from models.qwen2_5.modeling_qwen import Qwen2ForCausalLM
            model_cls = Qwen2ForCausalLM
        elif self.config.model_type == 't5':
            from models.t5.modeling_t5 import T5ForConditionalGeneration
            model_cls = T5ForConditionalGeneration
        else:
            raise ValueError(f"model_type:{self.config.model_type} is not defined yet.")

        if self.custom_args.load_func == "scratch":#Read the model configuration file and initialize the model structure
            config = AutoConfig.from_pretrained(checkpoint_path)
            model = model_cls(config)
        elif self.custom_args.load_func == "dense":#Only load dense
            # 1. Load parameters, randomly initialize the embedding layer
            model = model_cls.from_pretrained(checkpoint_path, device_map=device_map)   
            embed_layer = model.base_model.embed_tokens
            nn.init.normal_(embed_layer.weight, mean=0.0, std=0.02) ##randomly initialize
            # 2. freeze all layer
            for param in model.parameters():
                param.requires_grad = False
            # 3. Unfreeze the embedding layer (assuming the embedding layer is in model.base_model.embed_tokens)
            for param in embed_layer.parameters():
                param.requires_grad = True
        else:
            model = model_cls.from_pretrained(checkpoint_path, device_map=device_map)   
            
        #print('Old model: ', model)
        # expand model base new tokenizer
        tokenizer = self.tokenizer
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
        #print('New model: ', model)
        return model

    def create_dataset(self):
        dataset_name = self.config.dataset_name
        data_file = self.config.data_file
        if self.is_train:
            # 交替训练：原始数据用于 CE；额外数据用于 IAP+LDPO
            ldpo_data_file = getattr(self.custom_args, "ldpo_data_file", None)
            if ldpo_data_file:
                # 强制保留额外字段（否则 Trainer 会把 ldpo_* 列删掉，导致 loss 退化）
                self.training_args.remove_unused_columns = False

                print(f"📊 loading CE dataset...")
                if os.path.isfile(data_file):
                    dataset_ce = load_dataset("csv", data_files=data_file, split="train", streaming=self.config.streaming)
                else:
                    dataset_ce = load_dataset(dataset_name, data_files=data_file, split="train", streaming=self.config.streaming)

                print(f"📊 loading LDPO dataset...")
                if os.path.isfile(ldpo_data_file):
                    dataset_ldpo = load_dataset("csv", data_files=ldpo_data_file, split="train", streaming=self.config.streaming)
                else:
                    dataset_ldpo = load_dataset(dataset_name, data_files=ldpo_data_file, split="train", streaming=self.config.streaming)

                print("🔄 processing CE dataset...")
                preprocess_ce = self._create_preprocess_with_mode(training_mode=None)
                dataset_ce = dataset_ce.filter(preprocess_ce.filter_fn)
                tokenized_ce = dataset_ce.map(preprocess_ce, batched=False, remove_columns=["system", "user", "answer"])

                print("🔄 processing LDPO dataset...")
                preprocess_ldpo = self._create_preprocess_with_mode(training_mode="iap_ldpo")
                dataset_ldpo = dataset_ldpo.filter(preprocess_ldpo.filter_fn)
                tokenized_ldpo = dataset_ldpo.map(preprocess_ldpo, batched=False, remove_columns=["system", "user", "answer"])

                self.ldpo_train_dataset = tokenized_ldpo
                return tokenized_ce, None

            print(f"📊 loading dataset...")
            if os.path.isfile(data_file):
                dataset = load_dataset("csv", data_files=data_file, split="train", streaming=self.config.streaming)
            else:
                dataset = load_dataset(dataset_name, data_files=data_file, split="train", streaming=self.config.streaming)
            print("🔄 processing dataset...")
            dataset = dataset.filter(self.preprocess_function.filter_fn)
            tokenized_train = dataset.map(self.preprocess_function, batched=False, remove_columns=["system", "user", "answer"])
            return tokenized_train, None
        else:
            print(f"📊 loading dataset...")
            if os.path.isfile(data_file):
                dataset = load_dataset("csv", data_files=data_file, split="all")
            else:
                ## To read different files according to different stages
                dataset = load_dataset(dataset_name, data_files=data_file, split="all")
            print("🔄 processing dataset...")
            #tokenized_test = dataset["test"].map(self.preprocess_function, batched=False, remove_columns=["instruction", "input", "output"])
            dataset = dataset.filter(self.preprocess_function.filter_fn)
            tokenized_test = dataset.map(self.preprocess_function, batched=False)
            return None, tokenized_test

    def create_data_collator(self):
        # 交替训练：同时构建两套 collator，CE 用默认，LDPO 用专用 4D mask collator
        if getattr(self.custom_args, "ldpo_data_file", None):
            from utils.ldpo_data_collator import LDPODataCollator
            self.ldpo_data_collator = LDPODataCollator(
                tokenizer=self.tokenizer,
                padding=True,
                ldpo_only=bool(getattr(self.custom_args, "ldpo_only", False)),
            )
            return DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True)

        # IAP+LDPO 单一模式：使用自定义 collator，产出 4D item-aware attention mask
        if getattr(self.custom_args, "training_mode", None) == "iap_ldpo":
            from utils.ldpo_data_collator import LDPODataCollator
            data_collator = LDPODataCollator(
                tokenizer=self.tokenizer,
                padding=True,
                ldpo_only=bool(getattr(self.custom_args, "ldpo_only", False)),
            )
            return data_collator

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True)
        if not self.is_train:
            data_collator = DataCollatorWrapper(
                data_collator=data_collator,
                extra_feature_names=[
                    "id",
                    self.custom_args.instruction_column,
                    self.custom_args.input_column,
                    self.custom_args.output_column,
                ],
            )
        return data_collator


    def create_tokenlizer(self, tokenizer_path) -> Union[AutoTokenizer, None]:
        # add token.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        if self.is_train:
            print('Old tokenizer length: ', len(tokenizer))
            # add special token: [SEP]
            special_tokens_dict = {'additional_special_tokens': tokenizer.all_special_tokens + ['[SEP]']}
            tokenizer.add_special_tokens(special_tokens_dict)
            # add token: C0 ~ C65535
            tokenizer.add_tokens(['C%d' % i for i in range(0, 2 * 32768)])
            print('New tokenizer length: ', len(tokenizer))
        return tokenizer

    def create_trainer(self) -> Union[Trainer, None]:
        # create trainer
        trainer_cls = self.trainer_class()

        kwargs = {
            "model": self.model,
            "args": self.training_args,
            "train_dataset": self.train_dataset,
            "data_collator": self.data_collator,
            "tokenizer": self.tokenizer,
            "predict_output":self.predict_output
        }

        # 交替训练参数：如果提供了 LDPO 数据集，则传入 Trainer 以构建混合 dataloader
        if getattr(self.custom_args, "ldpo_data_file", None):
            kwargs["ldpo_train_dataset"] = self.ldpo_train_dataset
            kwargs["ldpo_data_collator"] = getattr(self, "ldpo_data_collator", None)
            kwargs["ldpo_ratio"] = float(getattr(self.custom_args, "ldpo_ratio", 0.5))
            kwargs["ldpo_seed"] = int(getattr(self.custom_args, "ldpo_seed", 42))
            kwargs["ldpo_start_with"] = str(getattr(self.custom_args, "ldpo_start_with", "ce"))

        ## loss function
        compute_loss_func = self.create_compute_loss_func()
        if compute_loss_func is not None:
            kwargs["compute_loss_func"] = compute_loss_func

        trainer = trainer_cls(**kwargs)
        return trainer

    def run(self,):
        self.tokenizer = self.create_tokenlizer(tokenizer_path = self.config.load_checkpoint_from)
        self.model = self.create_model()
        self.preprocess_function = self.create_preprocess()
        self.train_dataset, self.test_dataset = self.create_dataset()
        self.data_collator = self.create_data_collator()
        self.trainer = self.create_trainer()
        
        ## Execute training and testing according to the configuration file
        if self.training_args.do_train:
            self.trainer.train()
        elif self.training_args.do_predict:
            params = {"test_dataset": self.test_dataset}
            if self.gen_kwargs is not None:
                params.update(self.gen_kwargs)
            #It must be written to the file during each predict_step
            self.trainer.predict(**params)

    def close(self, success=True):
        # Do some finishing work,
        if self.trainer and self.trainer.predict_writer:
            return self.trainer.predict_writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    set_seed(42)

    ## python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 runner.py --config=config/t5_base_3layer.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    #Support distributed
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--gpu', default=0, type=int, help='')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help="Local ranking in distributed training. Automatically imported by PAI or XDL launcher")
    parser.add_argument('--dist_url', default='env://', help='Set the URL for distributed training')
    parser.add_argument('--distributed', action='store_true', help='Whether to enable distributed training')
    parser.add_argument('--device_type', default='', type=str, choices=['cuda', 'npu', ''],
                        help='Device: cuda or npu. Default from config or cuda.')
    args = parser.parse_args()

    config = EasyDict(args.config)
    ## Binding output address and configuration file name
    output_dir = config.training_args.get('output_dir', './logs/')
    config.training_args['output_dir'] = os.path.join(output_dir, os.path.splitext(os.path.basename(args.config))[0])
    if 'predict_output' in config:
        config.predict_output['path'] = config.training_args['output_dir']

    # device_type: 优先 CLI，其次 config，默认 cuda
    device_type = (args.device_type or getattr(config, 'device_type', None) or 'cuda').lower()
    config.device_type = device_type

    # Initialize the distributed environment (npu 用 hccl，cuda 用 nccl)
    dist_utils.init_distributed_mode(device_type, args)

    runner = Runner(config)
    success = True
    try:
        runner.run()
        logger.info("runner run success")
    except Exception as e:
        success = False
        logger.error("runner run failed, error=%s", traceback.format_exc())
    finally:
        try:
            runner.close(success=success)
        except:
            logger.warning("runner close failed, ignore, error=%s", traceback.format_exc())
    logger.info("run end")

if __name__ == '__main__':
    main()