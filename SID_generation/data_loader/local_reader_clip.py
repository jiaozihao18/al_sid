# encoding: utf-8
"""
@Author: Yingwu.XSW
@Date: 2024/7/30 上午11:05

@Function:
"""
from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def pad_dataset(dataset, global_batch_size):
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


class OSSFileImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, cfg, transform_func=None):
        super().__init__()
        # 读取文件
        data = np.load(root, allow_pickle=True)
        self.itemid2index, self.pairs, self.embeds = data['itemEncID'].item(), data['pairs'], data['embeds'].astype(
            np.float32)
        self.embeds_fuka = None

        self.pairs = self.pairs
        self.dataset_len = len(self.pairs)
        self.split = split
        self.transform = transform_func(self.split, cfg) if transform_func else None
        self.global_batch_size = 0
        self.cfg = cfg

        del data

    def __getitem__(self, i):
        rawid1, rawid2 = self.pairs[i]
        encid1, encid2 = self.itemid2index[rawid1], self.itemid2index[rawid2]
        return 0, self.embeds[encid1], 0, self.embeds[encid2]

    def __len__(self):
        return len(self.pairs)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler  # 可能是None（非分布式模式）
    dataset: OSSFileImageNetDataset
    epoch_id: int


def get_dataset(root, cfg, is_train, epoch_id=0):
    assert root is not None

    dataset = OSSFileImageNetDataset(
        root,
        split="train" if is_train else "val",
        cfg=cfg
    )

    batch_size = cfg.data.batch_size if is_train else cfg.data.valid_batch_size
    # 只在分布式模式下使用DistributedSampler
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

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

    if cfg.data.train_clip:
        if not dist.is_initialized() or dist.get_rank() == 0:  # 只在主进程打印
            print('preparing train_clip data...')
        data["train"] = get_dataset(cfg.data.train_clip, cfg, is_train=True, epoch_id=epoch_id)

    if len(cfg.data.train_clip2) > 1:
        if not dist.is_initialized() or dist.get_rank() == 0:  # 只在主进程打印
            print('preparing train_clip2 data...')
        data["train2"] = get_dataset(cfg.data.train_clip2, cfg, is_train=True, epoch_id=epoch_id)

    if len(cfg.data.train_clip3) > 1:
        if not dist.is_initialized() or dist.get_rank() == 0:  # 只在主进程打印
            print('preparing train_clip3 data...')
        data["train3"] = get_dataset(cfg.data.train_clip3, cfg, is_train=True, epoch_id=epoch_id)

    return data


def get_i2idata_list(path_list, cfg, epoch_id=0):
    data = {}

    for path in path_list:
        if not dist.is_initialized() or dist.get_rank() == 0:  # 只在主进程打印
            print(f'preparing {path} data...')
        data_name = path.split('/')[-1].split('.')[0]
        data[data_name] = get_dataset(path, cfg, is_train=True, epoch_id=epoch_id)

    return data
