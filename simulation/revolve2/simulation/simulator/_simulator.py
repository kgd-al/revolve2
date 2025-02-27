from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, auto
from typing import Callable

from ..scene import SimulationState
from ._batch import Batch


class Callback(Enum):
    START = auto()
    RENDER_START = auto()
    PRE_STEP = auto()
    PRE_CONTROL = auto()
    RENDER = auto()
    POST_CONTROL = auto()
    POST_STEP = auto()
    RENDER_END = auto()
    END = auto()


class Simulator(ABC):
    """Interface for a simulator."""

    def __init__(self):
        self._callbacks = defaultdict(list)

    @abstractmethod
    def simulate_batch(self, batch: Batch) -> list[list[SimulationState]]:
        """
        Simulate the provided batch by simulating each contained scene.

        :param batch: The batch to run.
        :returns: List of simulation states in ascending order of time.
        """
        pass

    def register_callback(self, ctype: Callback, callback: Callable):
        self._callbacks[ctype].append(callback)
