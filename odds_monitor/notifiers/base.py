"""Notifier interface: anything that can deliver a batch of discrepancies."""

from abc import ABC, abstractmethod
from typing import Iterable

from ..models import Discrepancy


class Notifier(ABC):
    @abstractmethod
    def notify(self, discrepancies: Iterable[Discrepancy]) -> None:
        raise NotImplementedError
