
from typing import Optional, Any, Dict, FrozenSet

from abc import ABC, abstractmethod

class MetricResult:
    """ For keeping result of evaluation

        Along with score can hold metadata of result like reasoning.
    """

    def __init__(self, value, 
                 reason: Optional[str] = None,
                 details: Optional[Dict[str, Any]]=None):

        self.value = value
        self.reason = reason
        self.details = details or {}


    def __repr__(self) -> str:
        return f"MetricResult(value={self.value!r}, reason={self.reason!r})"

    def __float__(self):
        return float(self.value)

    def __json__(self):
        return {
            "value": self.value,
            "reason" : self.reason
        }


class Metric(ABC):
    """Base class for all metrics in harness"""

    name: str
    required_inputs: FrozenSet[str] = frozenset()

    @abstractmethod
    def score(self, **kwargs)->MetricResult:
        """Compute the metric for 1 row of input"""
        ...
