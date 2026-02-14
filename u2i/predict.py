import os
import numpy as np
os.environ['NCCL_DEBUG'] = 'ERROR'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from options import args
from model import SASRec
from data.dataset import PreprocessedDataset, make_collate_fn
from datetime import datetime
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import faiss
from model import MODELS


@torch.no_grad()
def main_worker():

    print("开始加载模型...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MODELS[args.model_name](args.item_count, args, device=device)
    state_dict = torch.load(args.state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)  # 不需要 DDP 包装
    print("加载模型完成.")

    dataset_with_tensors = PreprocessedDataset(pt_path=args.data_path, is_train=False)

    dataloader = DataLoader(
        dataset_with_tensors,
        batch_size=50,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=make_collate_fn(is_train=False)
    )
    print("数据Load完成.")

    print("准备构建商品索引...")
    index = faiss.IndexFlatIP(args.hidden_units)
    index.add(model.item_emb.weight.data.cpu().numpy())
    print("商品索引构建完成...")

    Hit_20, Hit_100, Hit_500, Hit_1000 = [], [], [], []
    for batch in tqdm(dataloader):
        user_history, target_item = batch['user_history'], batch['target_item']
        final_feat = model.get_user_emb(user_history).cpu().numpy()
        _, indices_1000 = index.search(final_feat, 1000)  # [B, 1000]

        target_np = target_item.cpu().numpy()
        for i in range(final_feat.shape[0]):
            clicked = target_np[i]
            clicked_items = clicked[clicked != 0]
            if len(clicked_items) == 0:
                continue

            # 用 np.isin 向量化计算 hit
            n = len(clicked_items)
            hit_20 = np.isin(clicked_items, indices_1000[i, :20]).sum() / n
            hit_100 = np.isin(clicked_items, indices_1000[i, :100]).sum() / n
            hit_500 = np.isin(clicked_items, indices_1000[i, :500]).sum() / n
            hit_1000 = np.isin(clicked_items, indices_1000[i, :1000]).sum() / n
            Hit_20.append(hit_20)
            Hit_100.append(hit_100)
            Hit_500.append(hit_500)
            Hit_1000.append(hit_1000)

    print(f"Hit Rate_20: {sum(Hit_20)/len(Hit_20):.4f}")
    print(f"Hit Rate_100: {sum(Hit_100)/len(Hit_100):.4f}")
    print(f"Hit Rate_500: {sum(Hit_500)/len(Hit_500):.4f}")
    print(f"Hit Rate_1000: {sum(Hit_1000)/len(Hit_1000):.4f}")

if __name__ == "__main__":
    main_worker()