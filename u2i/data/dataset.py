import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
from torch.nn.utils.rnn import pad_sequence


class PreprocessedDataset(Dataset):
    """
    加载 process.py 预生成的 .pt 文件。
    负采样在 collate_fn 中按 batch 做，Dataset 只负责取数据。
    """
    def __init__(self, pt_path, is_train=True):
        self.data = torch.load(pt_path, map_location='cpu')
        self.user_history = self.data['user_history']  # [N, L, C]
        self.target_item = self.data['target_item']    # list of lists
        self.is_train = is_train

    def __len__(self):
        return len(self.target_item)

    def __getitem__(self, idx):
        user_history = self.user_history[idx]
        target_item = self.target_item[idx]
        if self.is_train:
            target_item = [target_item[0]] if target_item else [0]
        return {"user_history": user_history, "target_item": target_item}


def make_collate_fn(item_count=None, num_neg_samples=1, is_train=True):
    """
    返回 collate 函数。训练时在 collate 里按 batch 做负采样，一次向量化采样 [B, K]。
    """
    def collate(batch):
        uh0 = batch[0]["user_history"]
        if isinstance(uh0, torch.Tensor):
            user_history = torch.stack([item["user_history"] for item in batch])
        else:
            user_history = torch.tensor([item["user_history"] for item in batch], dtype=torch.long)

        target_item = [torch.tensor(item["target_item"], dtype=torch.long) for item in batch]
        target_item = pad_sequence(target_item, batch_first=True, padding_value=0)

        if is_train and item_count is not None and num_neg_samples > 0:
            B = len(batch)
            negative_items = torch.from_numpy(
                np.random.randint(1, item_count + 1, size=(B, num_neg_samples), dtype=np.int64)
            )
            return {"user_history": user_history, "target_item": target_item, "negative_items": negative_items}
        return {"user_history": user_history, "target_item": target_item}

    return collate


if __name__ == '__main__':
    # 测试 PreprocessedDataset + make_collate_fn
    pt_path = "/home/admin/workspace/aop_lab/data/AL-GR-Tiny/u2i/s1_tiny_with_feat.pt"
    dataset = PreprocessedDataset(pt_path=pt_path, is_train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=make_collate_fn(item_count=24573855, num_neg_samples=5, is_train=True)
    )
    for batch in dataloader:
        print(batch)
        break