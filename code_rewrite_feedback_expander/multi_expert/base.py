from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models import CodeRecord, RewriteCandidate, TokenDistribution
from .config import ExpertConfig


class ExpertCandidateGenerator(ABC):
    @abstractmethod
    def generate(self, expert: ExpertConfig, record: CodeRecord) -> RewriteCandidate:
        raise NotImplementedError


class ExpertTrajectoryScorer(ABC):
    @abstractmethod
    def score(
        self,
        expert: ExpertConfig,
        record: CodeRecord,
        candidate: RewriteCandidate,
    ) -> List[TokenDistribution]:
        raise NotImplementedError
