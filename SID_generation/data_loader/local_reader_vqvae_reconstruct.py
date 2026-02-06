# encoding: utf-8
"""
@author: Yingwu.XSW
@date: 2022/8/16 下午2:55
"""
import os.path
from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .local_reader_clip import get_i2idata_list


def pad_dataset(dataset, global_batch_size):
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


class OSSFileImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, cfg, transform_func=None):
        super().__init__()
        # 读取npz文件
        data = np.load(root, allow_pickle=True, mmap_mode='r')
        self.ids, self.embeds = data['ids'], data['embeds'].astype(np.float32)
        del data
        self.dataset_len = len(self.ids)

        self.split = split
        self.transform = transform_func(self.split, cfg) if transform_func else None
        self.global_batch_size = 0

    def __getitem__(self, i):
        _, embedding = self.ids[i], self.embeds[i]
        return 0, embedding.astype(np.float32)

    def __len__(self):
        return len(self.ids)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: OSSFileImageNetDataset
    epoch_id: int


def get_dataset(cfg, is_train, epoch_id=0):
    if is_train:
        root = cfg.data.train_root
    else:
        root = cfg.data.val_root
    assert root is not None

    dataset = OSSFileImageNetDataset(
        root,
        split="train" if is_train else "val",
        cfg=cfg
    )

    batch_size = cfg.data.batch_size if is_train else cfg.data.valid_batch_size
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        pin_memory=True,
        num_workers=cfg.data.num_workers if is_train else 1,
        sampler=sampler,
        drop_last=True
    )

    dataloader.num_samples = dataset.dataset_len
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(cfg, epoch_id=0):
    data = {}

    # 重建数据
    if not dist.is_initialized() or dist.get_rank() == 0:  # 只在主进程打印
        print('preparing recon data...')
    if cfg.data.train_root:
        data["recon"] = get_dataset(cfg, is_train=True, epoch_id=epoch_id)

    def update_data_with_conflict_check(data, new_data):
        # 检查重名的键
        conflicting_keys = set(new_data.keys()).intersection(set(data.keys()))

        if conflicting_keys:
            raise KeyError(f"Conflicting keys found: {conflicting_keys}")
        else:
            data.update(new_data)

    # 更新 i2i 数据
    i2i_data = get_i2idata_list(cfg.data.train_clip_i2i, cfg)
    update_data_with_conflict_check(data, i2i_data)

    return data
