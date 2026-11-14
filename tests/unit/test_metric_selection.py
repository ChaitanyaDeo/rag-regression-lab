"""
tests/unit/test_metric_selection.py

The metric catalog and the resolver behind `--metric`, `ragregress metrics`, GET /metrics and a dashboard.

  catalog contract   every REGISTERED metric is described well enough for a list or a UI, and can be built
                     by the resolver from what it declares (so a new metric cannot silently be unselectable)
  resolver           parsing, validation, build order, k and judge hand-over, lazy judge

No models, no database, no files.

Run (repo root):  python -m pytest tests/unit/test_metric_selection.py -v
"""

from __future__ import annotations

import pytest

from ragregress.core.errors import ConfigValidationError
from ragregress.evaluation.metrics import METRIC_REGISTRY
from ragregress.evaluation.metrics.base import COMMON_PARAMS, METRIC_GROUPS, Metric, MetricResult
from ragregress.evaluation.metrics.selection import (
    default_metric_names,
    metric_catalog,
    needs_judge,
    parse_metric_names,
    resolve_metrics,
    validate_metric_names,
)


def fake_judge():
    return object()


# ================================================================ catalog contract (real registry)

@pytest.mark.parametrize("name", list(METRIC_REGISTRY))
def test_every_registered_metric_is_fully_described(name):
    cls = METRIC_REGISTRY[name]
    assert cls.name == name
    assert cls.label.strip() and cls.description.strip()
    assert cls.group in METRIC_GROUPS
    assert cls.required_inputs                                         # score() inputs are declared
    assert cls.required_context <= COMMON_PARAMS                       # only things the resolver can hand over
    


@pytest.mark.parametrize("name", list(METRIC_REGISTRY))
def test_every_registered_metric_can_be_built_by_the_resolver(name):
    [metric] = resolve_metrics([name], top_k=5, get_judge=fake_judge)
    assert isinstance(metric, METRIC_REGISTRY[name])


def test_at_least_one_metric_is_on_by_default():
    assert default_metric_names()


def test_catalog_describes_gating_and_judge_cost():
    info = {i.name: i for i in metric_catalog()}
    assert list(info) == list(METRIC_REGISTRY)                         # registry order
    for name, cls in METRIC_REGISTRY.items():
        assert info[name].gated == (cls.regression_threshold is not None)
        assert info[name].threshold == cls.regression_threshold
        assert info[name].requires_judge == ("llm" in cls.required_context)


# ================================================================ resolver (tiny fake registry)

class _Plain(Metric):
    name, label, description, group = "plain", "Plain", "d", "answer"
    required_inputs = frozenset({"answer"})
    default_enabled = True

    def score(self, **kw):
        return MetricResult(value=1.0)


class _AtK(Metric):
    name, label, description, group = "at_k", "At k", "d", "retrieval"
    required_inputs = frozenset({"retrieved_ids"})
    required_context = frozenset({"k"})
    default_enabled = True

    def __init__(self, k):
        self.k = k

    def score(self, **kw):
        return MetricResult(value=1.0)


class _Judged(Metric):
    name, label, description, group = "judged", "Judged", "d", "generation"
    required_inputs = frozenset({"answer"})
    required_context = frozenset({"llm"})

    def __init__(self, llm):
        self.llm = llm

    def score(self, **kw):
        return MetricResult(value=1.0)


class _Judged2(_Judged):
    name = "judged2"


REG = {c.name: c for c in (_Plain, _AtK, _Judged, _Judged2)}


class Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return f"judge-{self.calls}"


def test_build_order_is_catalog_order_not_request_order():
    metrics = resolve_metrics(["judged", "plain", "at_k"], top_k=3, get_judge=fake_judge, registry=REG)
    assert [m.name for m in metrics] == ["plain", "at_k", "judged"]


def test_k_and_judge_are_handed_to_the_constructor():
    judge = Counter()
    metrics = {m.name: m for m in resolve_metrics(["at_k", "judged"], top_k=7, get_judge=judge, registry=REG)}
    assert metrics["at_k"].k == 7
    assert metrics["judged"].llm == "judge-1"


def test_judge_is_not_built_unless_a_selected_metric_needs_it():
    judge = Counter()
    resolve_metrics(["plain", "at_k"], top_k=3, get_judge=judge, registry=REG)
    assert judge.calls == 0


def test_judge_is_built_once_for_several_judged_metrics():
    judge = Counter()
    metrics = resolve_metrics(["judged", "judged2"], top_k=3, get_judge=judge, registry=REG)
    assert judge.calls == 1
    assert metrics[0].llm is metrics[1].llm


def test_defaults_are_the_metrics_flagged_default_enabled():
    assert default_metric_names(registry=REG) == ["plain", "at_k"]


def test_needs_judge():
    assert needs_judge(["plain", "judged"], registry=REG) is True
    assert needs_judge(["plain", "at_k"], registry=REG) is False


def test_unknown_required_context_is_a_programming_error():
    class Bad(_Plain):
        name = "bad"
        required_context = frozenset({"embedder"})

    with pytest.raises(ValueError, match="embedder"):
        resolve_metrics(["bad"], top_k=3, get_judge=fake_judge, registry={"bad": Bad})


# ---------------------------------------------------------------- validation

def problems(exc_info):
    return [(loc, msg) for loc, msg in exc_info.value.errors]


def test_unknown_duplicate_and_empty_selections_are_config_errors():
    with pytest.raises(ConfigValidationError) as e:
        validate_metric_names(["plain", "nope", "plain"], registry=REG)
    assert e.value.exit_code == 3
    assert problems(e) == [("'nope'", "unknown metric"), ("'plain'", "listed more than once")]
    assert "available: plain, at_k, judged, judged2" in str(e.value)

    with pytest.raises(ConfigValidationError) as e:
        validate_metric_names([], registry=REG)
    assert problems(e) == [("<selection>", "no metrics selected")]


def test_resolve_validates_before_building_anything():
    judge = Counter()
    with pytest.raises(ConfigValidationError):
        resolve_metrics(["judged", "nope"], top_k=3, get_judge=judge, registry=REG)
    assert judge.calls == 0


# ---------------------------------------------------------------- parsing the CLI text

def test_parse_splits_on_commas_and_trims_whitespace():
    assert parse_metric_names("plain, at_k ,judged", registry=REG) == ["plain", "at_k", "judged"]


def test_parse_all_means_every_registered_metric():
    assert parse_metric_names("all", registry=REG) == list(REG)
    assert parse_metric_names(" all ", registry=REG) == list(REG)


def test_parse_all_cannot_be_combined():
    with pytest.raises(ConfigValidationError) as e:
        parse_metric_names("all,plain", registry=REG)
    assert problems(e) == [("'all'", "cannot be combined with other names")]


@pytest.mark.parametrize("text", ["", "  ", "plain,,at_k", "plain,", ",plain"])
def test_parse_rejects_empty_items(text):
    with pytest.raises(ConfigValidationError) as e:
        parse_metric_names(text, registry=REG)
    assert "empty metric name" in problems(e)[0][1]


def test_parse_reports_unknown_names():
    with pytest.raises(ConfigValidationError) as e:
        parse_metric_names("plain,nope", registry=REG)
    assert problems(e) == [("'nope'", "unknown metric")]