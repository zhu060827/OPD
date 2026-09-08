# Copyright 2026 zhu060827/OPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic domain-balanced sampling for routed multi-Teacher OPD."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator

from verl.experimental.dataset.sampler import AbstractSampler


class DomainBalancedSampler(AbstractSampler):
    """Order indices so every consecutive generation batch follows target shares.

    Minority domains are sampled with replacement.  The remainder caused by a
    batch size that is not divisible by the number of domains rotates between
    domains across batches, avoiding a permanent first-domain preference.
    """

    def __init__(self, data_source, data_config):
        self.data_source = data_source
        self.batch_size = int(data_config.get("train_batch_size"))
        self.seed = int(data_config.get("seed", 0) or 0)
        sampler_config = data_config.get("sampler", {})
        self.domain_key = str(sampler_config.get("domain_key", "domain"))
        self.domains = tuple(
            sampler_config.get(
                "domains", ("cot", "style", "ast", "variable", "control_flow")
            )
        )
        raw_shares = sampler_config.get("target_shares", [1.0] * len(self.domains))
        if len(raw_shares) != len(self.domains):
            raise ValueError("target_shares must contain one value per domain")
        total = sum(float(value) for value in raw_shares)
        if self.batch_size < len(self.domains):
            raise ValueError("train_batch_size must be at least the number of domains")
        if total <= 0 or any(float(value) <= 0 for value in raw_shares):
            raise ValueError("all domain target shares must be positive")
        self.target_shares = tuple(float(value) / total for value in raw_shares)
        self.epoch = 0

        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is None or self.domain_key not in dataframe.column_names:
            raise ValueError(f"dataset must contain the {self.domain_key!r} column")
        self.indices_by_domain: dict[str, list[int]] = defaultdict(list)
        for index, domain in enumerate(dataframe[self.domain_key]):
            self.indices_by_domain[str(domain)].append(index)
        missing = [domain for domain in self.domains if not self.indices_by_domain[domain]]
        if missing:
            raise ValueError(f"training data is missing routed domains: {missing}")

    def __len__(self) -> int:
        return math.ceil(len(self.data_source) / self.batch_size) * self.batch_size

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        pools = {domain: rng.sample(indices, len(indices)) for domain, indices in self.indices_by_domain.items()}
        offsets = {domain: 0 for domain in self.domains}
        batch_count = len(self) // self.batch_size
        for batch_index in range(batch_count):
            counts = self._counts_for_batch(batch_index)
            batch: list[int] = []
            for domain, count in zip(self.domains, counts, strict=True):
                pool = pools[domain]
                for _ in range(count):
                    offset = offsets[domain]
                    if offset >= len(pool):
                        rng.shuffle(pool)
                        offset = 0
                    batch.append(pool[offset])
                    offsets[domain] = offset + 1
            rng.shuffle(batch)
            yield from batch
        self.epoch += 1

    def _counts_for_batch(self, batch_index: int) -> list[int]:
        exact = [share * self.batch_size for share in self.target_shares]
        counts = [math.floor(value) for value in exact]
        remainder = self.batch_size - sum(counts)
        priority = sorted(
            range(len(self.domains)),
            key=lambda index: (-(exact[index] - counts[index]), (index - batch_index) % len(self.domains)),
        )
        for index in priority[:remainder]:
            counts[index] += 1
        return counts
