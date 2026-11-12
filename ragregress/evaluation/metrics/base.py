
from typing import Optional, Any, Dict, FrozenSet

from abc import ABC, abstractmethod


METRIC_GROUPS = ("retrieval", "answer", "generation")
COMMON_PARAMS = frozenset({"k", "llm"})

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

    # Regression gate used during comparison of baseline vs candidate
    # Class-level on purpose: read from the CLASS (no instance needed; constructors take k / llm).
    # threshold = largest tolerated worsening, in this metric's own units. None = reported, not gated.

    regression_threshold: Optional[float] = None
    higher_is_better: Optional[bool] = True

    label: str = ""                                 # short display name
    description: str = ""                           # one sentence
    group: str = ""                                 # one of METRIC_GROUPS
    default_enabled: bool = False                   # selected when the user names no metrics
    required_context: FrozenSet[str] = frozenset()  # subset of COMMON_PARAMS: constructor arguments the harness supplies
    

    @abstractmethod
    def score(self, **kwargs)->MetricResult:
        """Compute the metric for 1 row of input"""
        ...
