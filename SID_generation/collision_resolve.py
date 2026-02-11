import argparse
import logging
from typing import List

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def resolve_collisions_with_candidates(
    item_ids: List[str],
    c1: np.ndarray,
    c2: np.ndarray,
    origin_c3: np.ndarray,
    candidates_c3: np.ndarray,
    max_per_bucket: int = 5,
) -> pd.DataFrame:
    """
    通用“阶段二”防碰撞逻辑：

    - 桶定义为完整 SID：bucket = (c1, c2, c3)。
      * 只有 c1、c2、c3 都相同，才算同一个桶。
    - 第一步：在每个原始桶 (c1, c2, origin_c3) 内随机保留最多 max_per_bucket 个。
    - 第二步：对剩余未分配的商品，按候选列表 candidates_c3 依次尝试：
      * c1、c2 不变，只替换 c3_candidate，形成新桶 (c1, c2, c3_candidate)，
        只要该桶当前计数 < max_per_bucket，就可以分配。
    - 第三步：如果所有候选都无法分配，则回退到原始桶 (c1, c2, origin_c3)，
      即使该桶已经超过 max_per_bucket（与 SQL / 现有实现的“极端兜底”行为保持一致）。
    """
    n_items = len(item_ids)
    if not (c1.shape[0] == c2.shape[0] == origin_c3.shape[0] == n_items):
        raise ValueError("c1, c2, origin_c3 的长度必须与 item_ids 一致。")

    if candidates_c3.ndim != 2 or candidates_c3.shape[0] != n_items:
        raise ValueError("candidates_c3 形状必须为 (N, K)，其中 N 与 item_ids 数量一致。")

    c1 = c1.astype(np.int64)
    c2 = c2.astype(np.int64)
    origin_c3 = origin_c3.astype(np.int64)
    candidates_c3 = candidates_c3.astype(np.int64)

    # 记录最终分配到的 c3（-1 表示尚未分配）
    assigned_c3 = np.full(n_items, -1, dtype=np.int64)

    # 以完整 SID (c1, c2, c3) 作为 key 计数：bucket_cap[(c1, c2, c3)] = 当前已分配个数
    bucket_cap: dict[tuple[int, int, int], int] = {}

    def _bucket_key(idx: int, c3_value: int) -> tuple[int, int, int]:
        return int(c1[idx]), int(c2[idx]), int(c3_value)

    # 第一步：在每个原始桶 (c1, c2, origin_c3) 内随机保留最多 max_per_bucket 个
    indices = np.arange(n_items)
    np.random.shuffle(indices)
    for i in indices:
        key = _bucket_key(i, origin_c3[i])
        cap = bucket_cap.get(key, 0)
        if cap < max_per_bucket:
            assigned_c3[i] = origin_c3[i]
            bucket_cap[key] = cap + 1

    # 第二步：按候选列表依次尝试，将尚未分配的商品尽量塞到未满的桶中
    k_candidates = candidates_c3.shape[1]
    for rank in range(k_candidates):
        unassigned = np.where(assigned_c3 < 0)[0]
        if unassigned.size == 0:
            break

        cand_codes = candidates_c3[unassigned, rank]
        for idx_in_un, item_idx in enumerate(unassigned):
            c3_cand = int(cand_codes[idx_in_un])
            key = _bucket_key(item_idx, c3_cand)
            cap = bucket_cap.get(key, 0)
            if cap >= max_per_bucket:
                continue
            assigned_c3[item_idx] = c3_cand
            bucket_cap[key] = cap + 1

    # 第三步：仍未能分配的，强制回退到原始桶 (c1, c2, origin_c3)
    still_unassigned = np.where(assigned_c3 < 0)[0]
    if still_unassigned.size > 0:
        logging.warning(
            "仍有 %d 个商品无法在候选列表容量限制内分配，回退使用原始 SID 桶（可能超过容量上限）。",
            still_unassigned.size,
        )
        for item_idx in still_unassigned:
            c3_value = int(origin_c3[item_idx])
            key = _bucket_key(item_idx, c3_value)
            cap = bucket_cap.get(key, 0)
            bucket_cap[key] = cap + 1
            assigned_c3[item_idx] = c3_value

    # 生成结果 DataFrame
    df = pd.DataFrame(
        {
            "item_id": item_ids,
            "c1": c1.astype(int),
            "c2": c2.astype(int),
            "origin_c3": origin_c3.astype(int),
            "assigned_c3": assigned_c3.astype(int),
        }
    )
    df["origin_SID"] = df[["c1", "c2", "origin_c3"]].astype(str).agg(",".join, axis=1)
    df["assigned_SID"] = df[["c1", "c2", "assigned_c3"]].astype(str).agg(",".join, axis=1)

    # 为每个完整桶 (c1, c2, assigned_c3) 内分配 rank（1..max_per_bucket），类似 SQL 里的 index
    df["rank"] = df.groupby(["c1", "c2", "assigned_c3"]).cumcount() + 1

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "阶段二：基于候选第 3 级码本 ID 的防碰撞分配脚本。\n"
            "桶定义为完整 SID (c1,c2,c3)，只有三者都相同才视为同一个桶。"
        )
    )
    parser.add_argument(
        "--candidate_csv",
        type=str,
        required=True,
        help=(
            "阶段一脚本生成的候选 CSV，需至少包含列：item_id, c1, c2, origin_c3, "
            "以及若干列候选 c3（列名形如 cand_1, cand_2, ...）。"
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="输出防碰撞后的结果 CSV 路径",
    )
    parser.add_argument(
        "--max_per_bucket",
        type=int,
        default=5,
        help="每个完整 SID 桶 (c1,c2,c3) 允许分配的最大商品数（与 SQL 中的 5 一致）",
    )
    return parser.parse_args()


def main() -> None:
    """
    使用示例（在 SID_generation 目录下）:

    阶段一（KNN 或随机）生成候选 CSV，例如:
      python collision_resolve_knn.py --mode knn   ...
      或
      python collision_resolve_knn.py --mode random ...

    阶段二（本脚本）做统一防碰撞:
      python collision_resolve.py \\
        --candidate_csv sorted_index_lv3_candidates.csv \\
        --output_csv collision_resolved.csv \\
        --max_per_bucket 5
    """
    args = parse_args()

    logging.info("读取候选 CSV: %s", args.candidate_csv)
    df = pd.read_csv(args.candidate_csv)

    required_cols = {"item_id", "c1", "c2", "origin_c3"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"候选 CSV 缺少必须列: {missing}")

    # 自动识别候选 c3 列：约定以 'cand_' 开头
    cand_cols = [c for c in df.columns if c.startswith("cand_")]
    if not cand_cols:
        raise ValueError("候选 CSV 中未找到任何候选列（列名需以 'cand_' 开头，例如 cand_1, cand_2, ...）。")

    logging.info("检测到 %d 个候选列: %s", len(cand_cols), ", ".join(cand_cols))

    item_ids = df["item_id"].astype(str).tolist()
    c1 = df["c1"].to_numpy()
    c2 = df["c2"].to_numpy()
    origin_c3 = df["origin_c3"].to_numpy()
    candidates_c3 = df[cand_cols].to_numpy()

    logging.info(
        "开始执行防碰撞分配逻辑：N=%d, 候选数 K=%d, 每个桶最多 %d 个商品...",
        len(item_ids),
        candidates_c3.shape[1],
        args.max_per_bucket,
    )
    result_df = resolve_collisions_with_candidates(
        item_ids=item_ids,
        c1=c1,
        c2=c2,
        origin_c3=origin_c3,
        candidates_c3=candidates_c3,
        max_per_bucket=args.max_per_bucket,
    )

    logging.info("写出结果到: %s", args.output_csv)
    result_df.to_csv(args.output_csv, index=False)
    logging.info("完成")


if __name__ == "__main__":
    main()

