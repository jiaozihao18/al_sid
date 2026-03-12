from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

import random


@dataclass
class AlternatingBatchIterable:
    """
    一个轻量 Iterable：交替从两个「已 collate 的 batch 迭代器」中取 batch。

    设计目标：
    - 每个 step 选择 CE 或 LDPO 的决策在所有 rank 上一致（避免 DDP 步数不一致）。
    - 当任一迭代器耗尽时自动重置（循环使用）。

    注意：
    - ce_iterable / ldpo_iterable 必须是可重复迭代的（每次 iter() 返回新的 iterator）。
    - 产出的元素应为 Trainer 可直接消费的 batch dict（tensor 在 CPU 上即可）。
    """

    ce_iterable: Iterable[Dict[str, Any]]
    ldpo_iterable: Iterable[Dict[str, Any]]
    ldpo_ratio: float = 0.5
    seed: int = 42
    start_with: str = "ce"  # "ce" or "ldpo"

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not (0.0 <= float(self.ldpo_ratio) <= 1.0):
            raise ValueError(f"ldpo_ratio must be in [0,1], got {self.ldpo_ratio}")

        ce_it = iter(self.ce_iterable)
        ldpo_it = iter(self.ldpo_iterable)

        # 用固定 seed 的 PRNG + step 计数，保证每个 rank 选择序列一致
        rng = random.Random(int(self.seed))
        step = 0

        # 一个小偏置：允许配置第一步从哪种 batch 开始
        force_first: Optional[bool] = None
        if self.start_with == "ce":
            force_first = False
        elif self.start_with == "ldpo":
            force_first = True

        while True:
            if force_first is not None and step == 0:
                take_ldpo = force_first
            else:
                take_ldpo = (rng.random() < float(self.ldpo_ratio))

            try:
                if take_ldpo:
                    batch = next(ldpo_it)
                else:
                    batch = next(ce_it)
            except StopIteration:
                # 迭代器耗尽则重置对应 iterator，再取一次
                if take_ldpo:
                    ldpo_it = iter(self.ldpo_iterable)
                    batch = next(ldpo_it)
                else:
                    ce_it = iter(self.ce_iterable)
                    batch = next(ce_it)

            step += 1
            yield batch

