"""Provider interface: anything that can hand back player-prop lines."""

from abc import ABC, abstractmethod
from typing import List

from ..models import PropLine


class OddsProvider(ABC):
    """Source of player-prop lines across sportsbooks for a given league."""

    @abstractmethod
    def fetch_player_props(self, league: str) -> List[PropLine]:
        """Return the current player-prop lines for the given league."""
        raise NotImplementedError
