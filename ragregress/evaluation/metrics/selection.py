"""
ragregress/evaluation/metrics/selection.py

Which metrics exist, which are on by default, and how a user's choice becomes metric objects.
One place for every entry point: `ragregress run --metric`, `ragregress metrics`, the API and a dashboard.

    metric_catalog()           what each metric is (label, group, judge cost, default, gated) -> for lists and UIs

    default_metric_names()     the metrics with `default_enabled`, in catalog order

    parse_metric_names(text)   "a,b" -> validated names; "all" = every registered metric

    validate_metric_names(l)   a list of names -> the same list, or a ConfigValidationError (exit 3 / HTTP 422)

    needs_judge(names)         True if any of them calls the judge (decides whether to check the judge settings)

    resolve_metrics(names, top_k=, get_judge=)
                               names -> metric objects, in catalog order


How a metric is built: it declares `required_context` (a subset of base.COMMON_PARAMS) and the resolver passes
those as constructor keywords: "k" -> top_k, "llm" -> the judge. `get_judge` is called at most once, and only
if a selected metric asks for "llm", so a run with no judged metric never builds a judge.
 
This module does not import ragregress.core.config (no import cycle); the registry is imported lazily.
 
"""

from __future__ import annotations
 
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence
 
from ragregress.core.errors import ConfigValidationError
 
ALL = "all"


@dataclass(frozen=True)
class MetricInfo:
    name: str
    label: str
    description: str
    group: str
    requires_judge: bool
    default_enabled: bool
    gated: bool
    threshold: Optional[float] = None
    higher_is_better: Optional[bool] = None

def _registry(registry: Optional[Mapping[str, type]]) -> Mapping[str, type]:
    if registry is None:
        from ragregress.evaluation.metrics import METRIC_REGISTRY as registry
    return registry


def _requires_judge(cls: type) -> bool:
    return "llm" in cls.required_context

def metric_catalog(registry: Optional[Mapping[str, type]] = None) -> List[MetricInfo]:
    return [
        MetricInfo(
            name=name,
            label=cls.label,
            description=cls.description,
            group=cls.group,
            requires_judge=_requires_judge(cls),
            default_enabled=cls.default_enabled,
            gated=cls.regression_threshold is not None,
            threshold=cls.regression_threshold,
            higher_is_better=cls.higher_is_better if cls.regression_threshold is not None else None,
        )
        for name, cls in _registry(registry).items()
    ]

def default_metric_names(registry: Optional[Mapping[str, type]] = None) -> List[str]:
    return [name for name, cls in _registry(registry).items() if cls.default_enabled]


def _selection_error(problems: List[tuple], registry: Mapping[str, type]) -> ConfigValidationError:
    return ConfigValidationError(
        f"invalid metric selection (available: {', '.join(registry)}, or '{ALL}')",
        errors=problems,
        source="metric",
    )
 


def validate_metric_names(names: Sequence[str], registry: Optional[Mapping[str, type]] = None) -> List[str]:
    reg = _registry(registry)
    problems: List[tuple] = []
 
    if not names:
        problems.append(("<selection>", "no metrics selected"))
 
    seen = set()
    for name in names:
        if name not in reg:
            problems.append((repr(name), "unknown metric"))
        elif name in seen:
            problems.append((repr(name), "listed more than once"))
        seen.add(name)
 
    if problems:
        raise _selection_error(problems, reg)
    return list(names)


def parse_metric_names(text: str, registry: Optional[Mapping[str, type]] = None) -> List[str]:
    reg = _registry(registry)
    items = [part.strip() for part in text.split(",")]
 
    if any(not item for item in items):
        raise _selection_error([("<selection>", "empty metric name (stray comma or empty value)")], reg)
 
    if ALL in items:
        if len(items) > 1:
            raise _selection_error([(repr(ALL), "cannot be combined with other names")], reg)
        return list(reg)
 
    return validate_metric_names(items, reg)


def needs_judge(names: Sequence[str], registry: Optional[Mapping[str, type]] = None) -> bool:
    reg = _registry(registry)
    return any(_requires_judge(reg[name]) for name in validate_metric_names(names, reg))



def resolve_metrics(names: Sequence[str], *, top_k: int, get_judge: Callable[[], Any],
                    registry: Optional[Mapping[str, type]] = None,) -> list:
    
    """Build the selected metrics, in catalog order (not the order the user typed them)."""
    reg = _registry(registry)
    wanted = set(validate_metric_names(names, reg))
 
    judge: List[Any] = []
 
    def judge_once() -> Any:
        if not judge:
            judge.append(get_judge())
        return judge[0]
 
    metrics = []
    for name, cls in reg.items():
        if name not in wanted:
            continue
        kwargs = {}
        for key in sorted(cls.required_context):
            if key == "k":
                kwargs["k"] = top_k
            elif key == "llm":
                kwargs["llm"] = judge_once()
            else:
                raise ValueError(f"metric {name!r} declares unknown required_context {key!r}")
        metrics.append(cls(**kwargs))
    return metrics