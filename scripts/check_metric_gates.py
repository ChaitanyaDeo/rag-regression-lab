
"""read each registered metric's regression gate from its CLASS (read-only).
 
Usage (repo root, venv active):
    python scripts/check_metric_gates.py
"""
from ragregress.evaluation.metrics import METRIC_REGISTRY
from ragregress.evaluation.metrics.base import Metric
 
 
def main():
    print(f"{'metric':<14}{'threshold':>10}  higher_is_better  class")
    problems = []
    for name, cls in METRIC_REGISTRY.items():
        thr = cls.regression_threshold
        print(f"{name:<14}{thr!s:>10}  {cls.higher_is_better!s:<16}  {cls.__name__}")
        if not issubclass(cls, Metric):
            problems.append(f"{name}: not a Metric subclass")
        if thr is not None and not (isinstance(thr, (int, float)) and thr > 0):
            problems.append(f"{name}: regression_threshold must be a positive number or None, got {thr!r}")
        if not isinstance(cls.higher_is_better, bool):
            problems.append(f"{name}: higher_is_better must be a bool")
    print(f"\nregistered: {len(METRIC_REGISTRY)}")
    print("problems:", problems or "(none)")
 
 
if __name__ == "__main__":
    main()