import argparse
import base64
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from infer_SID import EXPECTED_EMBEDDING_DIM, build_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def decode_embeddings_for_infer_rows(
    df_infer: pd.DataFrame,
    df_feat: pd.DataFrame,
    infer_id_col: str = "item_id",
    feat_id_col: str = "TODO",
    feat_emb_col: str = "TODO",
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
) -> Tuple[List[str], np.ndarray]:
    """
    按 infer_SID 输出的顺序，对应到原始特征表，解码得到 (N, D) 的 embedding。
    只保留在特征表中能找到的 item。
    """
    feat_id_series = df_feat[feat_id_col].astype(str)
    feat_emb_series = df_feat[feat_emb_col].astype(str)
    emb_map = dict(zip(feat_id_series.tolist(), feat_emb_series.tolist()))

    item_ids: List[str] = []
    embs: List[np.ndarray] = []

    missing = 0

    for item_id in df_infer[infer_id_col].astype(str).tolist():
        emb_str = emb_map.get(item_id)
        if emb_str is None:
            missing += 1
            continue
        try:
            vec = np.frombuffer(base64.b64decode(emb_str), dtype=np.float32)
            if vec.shape[0] != expected_dim:
                logging.warning(
                    "Item %s 的 embedding 维度异常: 期望 %d, 实际 %d, 已跳过",
                    item_id,
                    expected_dim,
                    vec.shape[0],
                )
                continue
            item_ids.append(item_id)
            embs.append(vec)
        except Exception as e:  # noqa: BLE001
            logging.warning("Item %s 解码失败: %s, 已跳过", item_id, e)

    if missing > 0:
        logging.warning("有 %d 个 item_id 在特征表中未找到，将被跳过。", missing)

    if not item_ids:
        raise RuntimeError("没有成功解码任何 embedding，请检查特征表与 inference 结果的 item_id 是否对齐。")

    return item_ids, np.stack(embs, axis=0)


def parse_sid_column_for_items(
    df_infer: pd.DataFrame,
    kept_item_ids: List[str],
    id_col: str = "item_id",
    sid_col: str = "SID",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对照 kept_item_ids（embedding 成功解码的 item），取出对应的 SID，并拆成 c1,c2,c3。
    顺序与 kept_item_ids 一一对应。
    """
    infer_map = dict(
        zip(
            df_infer[id_col].astype(str).tolist(),
            df_infer[sid_col].astype(str).tolist(),
        )
    )

    sids: List[str] = []
    for item_id in kept_item_ids:
        sid = infer_map.get(item_id)
        if sid is None:
            raise RuntimeError(f"inference 结果中缺少 item_id={item_id} 的 SID。")
        sids.append(sid)

    parts = pd.Series(sids).str.split(",", expand=True)
    if parts.shape[1] != 3:
        raise ValueError(f"SID 列拆分后列数为 {parts.shape[1]}，预期为 3，请检查 SID 格式。")

    parts = parts.astype(int)
    c1 = parts[0].to_numpy()
    c2 = parts[1].to_numpy()
    c3 = parts[2].to_numpy()
    return c1, c2, c3


def compute_codes_and_sorted_indices(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    batch_size: int = 1024,
    k_candidates: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用模型计算：
    - codes: (N, 3) 三层码本 ID
    - sorted_indices: (N, K) 第 3 层码本的 KNN 候选（按距离从小到大排序，去掉第一个自身）
    """
    device = next(model.parameters()).device

    all_codes: List[np.ndarray] = []
    all_sorted: List[np.ndarray] = []

    total = embeddings.shape[0]
    k_candidates = int(k_candidates)

    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc="Computing codes & KNN (level 3)"):
            end = min(start + batch_size, total)
            batch_np = embeddings[start:end]
            x = torch.from_numpy(batch_np).to(device)

            # 使用 RQVAE_EMBED_CLIP.get_sorted_index
            code, sorted_indices = model.get_sorted_index(x)
            code_np = code.cpu().numpy()  # (B, 3)
            sorted_np = sorted_indices.cpu().numpy()  # (B, codebook_size)

            # 跟 SQL 脚本一致：去掉第一个（通常是自身），只保留 2..K+1
            if sorted_np.shape[1] <= 1:
                raise RuntimeError("sorted_indices 的列数过小，无法构造 KNN 列表。")
            sorted_np = sorted_np[:, 1 : k_candidates + 1]

            all_codes.append(code_np)
            all_sorted.append(sorted_np)

    codes = np.concatenate(all_codes, axis=0)
    sorted_indices = np.concatenate(all_sorted, axis=0)
    return codes, sorted_indices


def generate_random_candidates(
    origin_c3: np.ndarray,
    codebook_size: int,
    k_candidates: int,
) -> np.ndarray:
    """
    生成随机候选的第 3 层码本 ID 列表，形状为 (N, K)。
    每个位置独立从 [0, codebook_size) 均匀采样。
    """
    n_items = origin_c3.shape[0]
    k_candidates = int(k_candidates)
    if codebook_size <= 0:
        raise ValueError("codebook_size 必须为正整数。")
    # 可以选择是否排除 origin_c3；这里保持简单实现：允许与 origin_c3 相同。
    return np.random.randint(0, codebook_size, size=(n_items, k_candidates), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "阶段一：生成防碰撞用的候选第 3 级码本 ID（KNN 或随机），"
            "输出统一的 candidate CSV，供 collision_resolve.py 使用。"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="knn",
        choices=["knn", "random"],
        help="候选生成模式：knn 基于模型 KNN，random 为纯随机。",
    )
    parser.add_argument(
        "--infer_csv",
        type=str,
        required=True,
        help="infer_SID.py 生成的 CSV，需包含列: item_id, SID",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="输出候选 CSV 路径（供阶段二 collision_resolve.py 使用）",
    )

    # 以下参数仅在 mode=knn 时需要
    parser.add_argument(
        "--feature_csv",
        type=str,
        help="原始 embedding 特征 CSV（infer_SID 的输入），需包含 item_id 和 embedding 列，仅在 mode=knn 时使用",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="训练好的模型 checkpoint 路径（与 infer_SID 使用的一致，仅在 mode=knn 时使用）",
    )
    parser.add_argument(
        "--infer_id_col",
        type=str,
        default="item_id",
        help="infer_SID 输出表中的 item_id 列名",
    )
    parser.add_argument(
        "--infer_sid_col",
        type=str,
        default="SID",
        help="infer_SID 输出表中的 SID 列名",
    )
    parser.add_argument(
        "--feat_id_col",
        type=str,
        default="TODO",
        help="特征表中的 item_id 列名（与 infer 输入一致，仅在 mode=knn 时使用）",
    )
    parser.add_argument(
        "--feat_emb_col",
        type=str,
        default="TODO",
        help="特征表中的 embedding(base64) 列名（与 infer 输入一致，仅在 mode=knn 时使用）",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default="cuda",
        choices=["cuda", "npu", "cpu"],
        help="推理设备类型（需与训练一致，仅在 mode=knn 时使用）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="计算 KNN 时的 batch size（仅在 mode=knn 时使用）",
    )
    parser.add_argument(
        "--k_candidates",
        type=int,
        default=200,
        help="每个商品使用的候选数量 K（不含自身）",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=8192,
        help="第 3 级码本的大小（默认为 8192，仅 random 模式需要用于采样范围）",
    )
    return parser.parse_args()


def main() -> None:
    """
    使用示例（在 SID_generation 目录下）:

    1）KNN 模式生成候选:
        python collision_resolve_knn.py \\
          --mode knn \\
          --infer_csv inference_results_batch.csv \\
          --feature_csv ./item_feature/final/part_01.csv \\
          --ckpt_path output_model/checkpoint-7.pth \\
          --feat_id_col 实际ID列名 \\
          --feat_emb_col 实际Embedding列名 \\
          --device_type cuda \\
          --batch_size 1024 \\
          --k_candidates 200 \\
          --output_csv sorted_index_lv3_candidates_knn.csv

    2）随机模式生成候选（不依赖 embedding / 模型）:
        python collision_resolve_knn.py \\
          --mode random \\
          --infer_csv inference_results_batch.csv \\
          --k_candidates 200 \\
          --codebook_size 8192 \\
          --output_csv sorted_index_lv3_candidates_random.csv
    """
    args = parse_args()

    logging.info("读取 infer_SID 输出: %s", args.infer_csv)
    df_infer = pd.read_csv(args.infer_csv)
    if args.infer_id_col not in df_infer.columns or args.infer_sid_col not in df_infer.columns:
        raise ValueError(
            f"infer_SID 输出需要包含列: {args.infer_id_col}, {args.infer_sid_col}"
        )

    # 解析 SID，得到 c1, c2, c3
    infer_item_ids = df_infer[args.infer_id_col].astype(str).tolist()
    c1_all, c2_all, c3_all = parse_sid_column_for_items(
        df_infer=df_infer,
        kept_item_ids=infer_item_ids,
        id_col=args.infer_id_col,
        sid_col=args.infer_sid_col,
    )

    mode = args.mode
    logging.info("候选生成模式: %s", mode)

    if mode == "knn":
        # KNN 模式需要 embedding 与模型
        if not args.feature_csv or not args.ckpt_path:
            raise ValueError("mode=knn 时必须提供 --feature_csv 和 --ckpt_path。")

        logging.info("读取原始特征表: %s", args.feature_csv)
        df_feat = pd.read_csv(args.feature_csv)
        for col in [args.feat_id_col, args.feat_emb_col]:
            if col not in df_feat.columns:
                raise ValueError(f"特征表中缺少列: {col}")

        logging.info("对齐 infer 结果与特征表，开始解码 embedding...")
        item_ids, embeddings = decode_embeddings_for_infer_rows(
            df_infer=df_infer,
            df_feat=df_feat,
            infer_id_col=args.infer_id_col,
            feat_id_col=args.feat_id_col,
            feat_emb_col=args.feat_emb_col,
            expected_dim=EXPECTED_EMBEDDING_DIM,
        )
        logging.info("成功对齐并解码 %d 条样本的 embedding。", len(item_ids))

        # 对齐后的 item_ids 顺序可能是 infer 的子集，需同步 c1, c2, c3
        id_to_index = {str(iid): idx for idx, iid in enumerate(infer_item_ids)}
        indices = []
        for iid in item_ids:
            idx = id_to_index.get(str(iid))
            if idx is None:
                raise RuntimeError(f"在 infer_SID 结果中找不到 item_id={iid} 的 SID。")
            indices.append(idx)
        c1 = c1_all[indices]
        c2 = c2_all[indices]
        c3 = c3_all[indices]

        origin_codes = np.stack([c1, c2, c3], axis=1)

        logging.info("构建模型并加载权重（用于计算第 3 级码本的 KNN）...")
        model = build_model(args.ckpt_path, device_type=args.device_type)

        logging.info("开始计算 codes 以及第 3 级码本的 KNN 候选...")
        codes_from_model, knn_indices = compute_codes_and_sorted_indices(
            model=model,
            embeddings=embeddings,
            batch_size=args.batch_size,
            k_candidates=args.k_candidates,
        )

        # 可选：检查 infer_SID 输出的 code 与当前模型计算的一致性
        if not np.array_equal(origin_codes, codes_from_model):
            diff_count = np.sum(origin_codes != codes_from_model)
            logging.warning(
                "origin_codes 与当前模型计算的 codes 不完全一致，有 %d 个元素不同，"
                "将继续使用 infer_SID 的 origin_codes 作为原始 SID。",
                int(diff_count),
            )

        item_ids_final = item_ids
        c1_final, c2_final, c3_final = c1, c2, c3
        candidates_c3 = knn_indices.astype(np.int64)

    else:  # mode == "random"
        # 随机模式：不需要特征表和模型，直接对 infer 中的所有 item 生成随机候选
        item_ids_final = infer_item_ids
        c1_final, c2_final, c3_final = c1_all, c2_all, c3_all
        logging.info(
            "使用随机模式生成候选：N=%d, K=%d, codebook_size=%d",
            len(item_ids_final),
            args.k_candidates,
            args.codebook_size,
        )
        candidates_c3 = generate_random_candidates(
            origin_c3=c3_final,
            codebook_size=args.codebook_size,
            k_candidates=args.k_candidates,
        )

    # 生成统一格式的候选 CSV：item_id, c1, c2, origin_c3, cand_1..cand_K
    k = candidates_c3.shape[1]
    data = {
        "item_id": item_ids_final,
        "c1": c1_final.astype(int),
        "c2": c2_final.astype(int),
        "origin_c3": c3_final.astype(int),
    }
    for j in range(k):
        data[f"cand_{j + 1}"] = candidates_c3[:, j].astype(int)

    df_out = pd.DataFrame(data)
    logging.info("写出候选 CSV 到: %s", args.output_csv)
    df_out.to_csv(args.output_csv, index=False)
    logging.info("完成（模式: %s，候选数 K=%d，样本数 N=%d）", mode, k, len(item_ids_final))


if __name__ == "__main__":
    main()

