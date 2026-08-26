"""Notifier interface: anything that can deliver a batch of discrepancies."""

from abc import ABC, abstractmethod
from typing import Iterable

from ..models import Alertable


class Notifier(ABC):
    @abstractmethod
    def notify(self, discrepancies: Iterable[Alertable]) -> None:
        raise NotImplementedError
