import pandas as pd
import numpy as np
import random
import os
import torch
from tqdm import tqdm

item_mapping = {}
feat_dict = {}  # sid -> (lv1, lv2, lv3)，从 tiny_item_sid_base.csv 直接读取，不生成 JSON
cnt = 1

def _read_csv(path):
    """读取 CSV，优先使用 pyarrow 引擎加速"""
    try:
        return pd.read_csv(path, engine='pyarrow')
    except (ImportError, ValueError):
        return pd.read_csv(path)

def load_feat_from_csv(csv_path):
    """从 tiny_item_sid_base.csv 直接读取 feat，不生成 tiny_sid_base_dict.json（向量化）"""
    global feat_dict
    df = _read_csv(csv_path)
    # 向量化 lv 转换
    def _lv_col(col):
        mask = pd.notna(col) & (col.astype(str).str.strip() != '')
        out = np.zeros(len(col), dtype=np.int64)
        out[mask] = col[mask].astype(float).astype(int).values + 1
        return out
    lv1, lv2, lv3 = _lv_col(df['codebook_lv1']), _lv_col(df['codebook_lv2']), _lv_col(df['codebook_lv3'])
    feat_dict.update(zip(df['base62_string'], zip(lv1, lv2, lv3)))

def process(data):
    global item_mapping, cnt  # 声明为全局变量
    for row in data.itertuples():
        items = row.user_history.split(';')
        for item in items:
            if item not in item_mapping:
                item_mapping[item] = cnt
                cnt += 1
        target_items = row.target_item.split(';')
        for item in target_items:
            if item not in item_mapping:
                item_mapping[item] = cnt
                cnt += 1

def process_raw_to_pt(df_or_path, pt_path, max_length, with_feat=True):
    """
    从 DataFrame 或 CSV 路径生成 .pt 文件。接收 df 可避免重复读取。
    with_feat=True: [id, lv1, lv2, lv3], feat_dim=4，供 sasrec_addfeat 等
    with_feat=False: [id], feat_dim=1，供 basemodel/sasrec/bert4rec/hstu 等
    """
    global item_mapping, feat_dict
    df = df_or_path if isinstance(df_or_path, pd.DataFrame) else _read_csv(df_or_path)
    n = len(df)
    feat_dim = 4 if with_feat else 1
    user_history_arr = np.zeros((n, max_length, feat_dim), dtype=np.int64)
    target_item_list = []
    default_feat = (0, 0, 0)
    get_mapping = item_mapping.get
    get_feat = feat_dict.get

    desc = f"Processing {os.path.basename(pt_path)}"
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=n, desc=desc)):
        hist_str = row.user_history
        if pd.isna(hist_str) or str(hist_str).strip() == '':
            target_item_list.append([0])
            continue

        sids = str(hist_str).strip().split(';')[::-1]
        if with_feat:
            seq = np.array([[get_mapping(sid, 0)] + list(get_feat(sid, default_feat)) for sid in sids], dtype=np.int64)
        else:
            seq = np.array([[get_mapping(sid, 0)] for sid in sids], dtype=np.int64)
        L = len(seq)
        if L < max_length:
            user_history_arr[i, :L] = seq
            # user_history_arr[i, L:] 已零初始化，无需 pad
        else:
            user_history_arr[i] = seq[-max_length:]

        tgt_str = row.target_item
        if pd.isna(tgt_str) or str(tgt_str).strip() == '':
            target_item_list.append([0])
        else:
            target_item_list.append([get_mapping(s, 0) for s in str(tgt_str).strip().split(';')])

    user_history_tensor = torch.from_numpy(user_history_arr)
    data_dict = {
        'user_history': user_history_tensor,
        'target_item': target_item_list,
        'max_length': max_length,
        'feat_dim': feat_dim,
    }
    torch.save(data_dict, pt_path)
    print(f"✅ 已保存到 {pt_path} (shape: {user_history_tensor.shape})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/home/admin/workspace/aop_lab/data/AL-GR-Tiny", type=str)
    parser.add_argument('--max_length', default=100, type=int, help='与 options.maxlen 一致，用于 .pt 预处理的序列长度')
    _args = parser.parse_args()
    data_path = _args.data_path
    print("load data (single read per file)")
    df_train = _read_csv(os.path.join(data_path, 'origin_behavior/s1_tiny.csv'))
    df_test = _read_csv(os.path.join(data_path, 'origin_behavior/s1_tiny_test.csv'))

    print("build item mapping")
    process(df_train)
    process(df_test)

    items = list(item_mapping.items())
    random.shuffle(items)  # 随机打乱 item 和 id 的顺序
    item_mapping = {k: v + 1 for v, (k, _) in enumerate(items)}  # 重新分配 id

    feat_csv = os.path.join(data_path, 'item_info/tiny_item_sid_base.csv')
    if os.path.isfile(feat_csv):
        print("load feat from tiny_item_sid_base.csv")
        load_feat_from_csv(feat_csv)
    else:
        print("未找到 tiny_item_sid_base.csv，feat 将全为 0")

    print("start process (raw -> .pt)")
    u2i_dir = os.path.join(data_path, 'u2i')
    os.makedirs(u2i_dir, exist_ok=True)

    max_length = _args.max_length
    # 生成 4 个文件：train/test × with_feat/without_feat，供不同模型使用
    process_raw_to_pt(df_train, os.path.join(u2i_dir, 's1_tiny_with_feat.pt'), max_length=max_length, with_feat=True)
    process_raw_to_pt(df_train, os.path.join(u2i_dir, 's1_tiny_without_feat.pt'), max_length=max_length, with_feat=False)
    process_raw_to_pt(df_test, os.path.join(u2i_dir, 's1_tiny_test_with_feat.pt'), max_length=max_length, with_feat=True)
    process_raw_to_pt(df_test, os.path.join(u2i_dir, 's1_tiny_test_without_feat.pt'), max_length=max_length, with_feat=False)