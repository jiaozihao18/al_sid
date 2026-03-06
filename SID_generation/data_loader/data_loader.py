# encoding: utf-8
"""
Embedding 数据加载：mode='recon' 单条重建，mode='i2i' 成对对比。
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class EmbeddingDataset(Dataset):
    """
    mode='recon': 含 ids, embeds，返回 (0, embedding)。
    mode='i2i': 含 itemEncID, pairs, embeds，返回 (0, embed1, 0, embed2)。
    """

    def __init__(self, root, mode, split, cfg, transform_func=None):
        super().__init__()
        self.mode = mode
        self.split = split
        self.transform = transform_func(split, cfg) if transform_func else None
        self.global_batch_size = 0
        self.cfg = cfg

        data = np.load(root, allow_pickle=True, mmap_mode='r' if mode == 'recon' else None)
        if mode == 'recon':
            self.ids = data['ids']
            self.embeds = data['embeds'].astype(np.float32)
            self.dataset_len = len(self.ids)
        else:
            self.itemid2index = data['itemEncID'].item()
            self.pairs = data['pairs']
            self.embeds = data['embeds'].astype(np.float32)
            self.dataset_len = len(self.pairs)
        del data

    def __getitem__(self, i):
        if self.mode == 'recon':
            return 0, self.embeds[i].astype(np.float32)
        rawid1, rawid2 = self.pairs[i]
        encid1, encid2 = self.itemid2index[rawid1], self.itemid2index[rawid2]
        return 0, self.embeds[encid1], 0, self.embeds[encid2]

    def __len__(self):
        return self.dataset_len


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler  # 可能是None（非分布式模式）
    dataset: EmbeddingDataset


def get_dataset(root, cfg, is_train, mode='recon'):
    assert root is not None
    dataset = EmbeddingDataset(
        root,
        mode=mode,
        split="train" if is_train else "val",
        cfg=cfg,
    )
    batch_size = cfg.data.batch_size if is_train else cfg.data.valid_batch_size
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
        drop_last=True,
    )
    dataloader.num_samples = dataset.dataset_len
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler, dataset)


def get_i2idata_list(path_list, cfg):
    """i2i 对比数据：多个路径，每个用 mode='i2i' 建 DataInfo。"""
    data = {}
    for path in path_list:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f'preparing {path} data...')
        data_name = path.split('/')[-1].split('.')[0]
        data[data_name] = get_dataset(path, cfg, is_train=True, mode='i2i')
    return data


def get_data(cfg):
    data = {}
    if not dist.is_initialized() or dist.get_rank() == 0:
        print('preparing recon data...')
    if cfg.data.train_root:
        data["recon"] = get_dataset(cfg.data.train_root, cfg, is_train=True, mode='recon')

    def update_data_with_conflict_check(data, new_data):
        conflicting_keys = set(new_data.keys()).intersection(set(data.keys()))
        if conflicting_keys:
            raise KeyError(f"Conflicting keys found: {conflicting_keys}")
        data.update(new_data)

    i2i_data = get_i2idata_list(cfg.data.train_clip_i2i, cfg)
    update_data_with_conflict_check(data, i2i_data)
    return data
