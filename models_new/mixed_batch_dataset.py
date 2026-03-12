from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

import random


@dataclass
class AlternatingBatchIterable:
    """在两个 batch 迭代器之间按比例交替采样。"""
    ce_iterable: Iterable[Dict[str, Any]]
    ldpo_iterable: Iterable[Dict[str, Any]]
    ldpo_ratio: float = 0.5
    seed: int = 42
    start_with: str = "ce"

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not (0.0 <= float(self.ldpo_ratio) <= 1.0):
            raise ValueError(f"ldpo_ratio must be in [0,1], got {self.ldpo_ratio}")

        ce_it = iter(self.ce_iterable)
        ldpo_it = iter(self.ldpo_iterable)

        rng = random.Random(int(self.seed))
        step = 0

        force_first: Optional[bool] = None
        if self.start_with == "ce":
            force_first = False
        elif self.start_with == "ldpo":
            force_first = True

        while True:
            if force_first is not None and step == 0:
                take_ldpo = force_first
            else:
                take_ldpo = rng.random() < float(self.ldpo_ratio)

            try:
                if take_ldpo:
                    batch = next(ldpo_it)
                else:
                    batch = next(ce_it)
            except StopIteration:
                if take_ldpo:
                    ldpo_it = iter(self.ldpo_iterable)
                    batch = next(ldpo_it)
                else:
                    ce_it = iter(self.ce_iterable)
                    batch = next(ce_it)

            step += 1
            yield batch

