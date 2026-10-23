"""
tests/unit/test_report.py

ragregress/report.py: context building (pure) and Jinja2 rendering, against a temporary SQLite
database plus temporary passages / split files. No LLM, no pipeline run, no real template
(render tests write a tiny template into tmp_path, so this file does not depend on D9-B4).

Run (repo root):  python -m pytest tests/unit/test_report.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from jinja2.exceptions import UndefinedError

from ragregress.core.errors import UnknownRunError
from ragregress.evaluation.comparator import MISSING
from ragregress.evaluation.regression import Gate
from ragregress.report import (
    DASH,
    NO_TARGETS_NOTE,
    TargetsFileError,
    default_output_path,
    fmt,
    fmt_config_value,
    fmt_ms,
    fmt_signed,
    generate_report,
    load_targets,
    render_html,
    target_status,
    write_report,
)
from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

M4 = ("recall_at_k", "mrr", "token_prf1", "faithfulness")
GATES = {name: Gate(0.05, True) for name in M4}
NOW = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- fixtures / builders

@pytest.fixture()
def repo(tmp_path) -> Repository:
    db = tmp_path / "test_report.db"
    init_db(db)
    return Repository(db)


@pytest.fixture()
def data_files(tmp_path):
    passages = tmp_path / "passages.json"
    passages.write_text(
        json.dumps([
            {"passage_id": "p1", "title": "T1", "text": "Passage one text."},
            {"passage_id": "p2", "title": "T2", "text": "Passage two."},
        ]),
        encoding="utf-8",
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "val.json").write_text(
        json.dumps([{"qa_id": f"q{i}", "question": f"Q{i}?", "answer": [f"gold {i}"]} for i in range(10)]),
        encoding="utf-8",
    )
    return passages, splits


DEC03_YAML = """\
source: README.md DEC-03 (2026-09-10)
targets:
  recall_at_k:  {label: "Recall@5", low: 0.80, high: 0.85}
  mrr:          {label: "MRR", low: 0.65, high: 0.75}
  token_prf1:   {label: "Answer F1", low: 0.55, high: 0.65}
  faithfulness: {label: "Faithfulness", low: 0.80, high: 0.90}
"""


@pytest.fixture()
def targets_file(tmp_path):
    path = tmp_path / "targets.yaml"
    path.write_text(DEC03_YAML, encoding="utf-8")
    return path


def _config(chunk: int = 500) -> dict:
    return {
        "pipeline": {"name": "builtin_rag", "params": {"chunk_size": chunk}},
        "eval_split": "val",
        "judge_model": "judge-7b",
        "is_dev": False,
    }


def add_run(repo, run_id, rows, *, chunk=500, latency=None, config_hash=None):
    repo.create_run(
        Run(run_id=run_id, created_at="2026-10-23T00:00:00", is_dev=False, k=5, max_workers=1,
            config=_config(chunk), config_hash=config_hash or f"hash-{run_id}")
    )
    repo.save_query_results(rows)
    if latency:
        repo.save_metrics(run_id, dict(latency))


def qr(run_id, qa_id, *, status="judged", answer="an answer", retrieved=("p1",), gold=("p1",),
       category="simple", failure_mode="pass", reason="all claims supported", **values) -> QueryResult:
    metric_results = {
        name: {"value": v, "reason": reason if name == "faithfulness" else None} for name, v in values.items()
    }
    return QueryResult(
        run_id=run_id, qa_id=qa_id, question=f"Q{qa_id[1:]}?", query_status=status,
        failure_mode=failure_mode, last_error=None, retrieved_ids=list(retrieved), gold_ids=list(gold),
        generation_answer=answer, metric_results=metric_results, latency={}, category=category,
    )


def full(v: float = 1.0) -> dict:
    return {name: v for name in M4}


def build(repo, data_files, base_rows, cand_rows, *, base_latency=None, cand_latency=None,
          passages_path=None, splits_dir=None, top=5, targets_path=None):
    add_run(repo, "base", base_rows, latency=base_latency)
    add_run(repo, "cand", cand_rows, chunk=1000, latency=cand_latency)
    passages, splits = data_files
    return generate_report(
        repo, "base", "cand",
        passages_path=passages_path or passages,
        splits_dir=splits_dir or splits,
        top=top, targets_path=targets_path, gates=GATES, now=NOW,
    )


def same_rows(n=5, v=1.0):
    return ([qr("base", f"q{i}", **full(v)) for i in range(n)],
            [qr("cand", f"q{i}", **full(v)) for i in range(n)])


# ---------------------------------------------------------------- formatting helpers

def test_number_formatting():
    assert fmt(None) == DASH
    assert fmt(float("nan")) == DASH
    assert fmt(0.71565) == "0.7157"
    assert fmt_signed(0.04) == "+0.0400"
    assert fmt_signed(-0.0267) == "-0.0267"
    assert fmt_signed(None) == DASH


def test_latency_and_config_formatting():
    assert fmt_ms(20083.4) == "20,083 ms"
    assert fmt_ms(500.0, signed=True) == "+500 ms"
    assert fmt_ms(None, signed=True) == DASH
    assert fmt_config_value(MISSING) == DASH
    assert fmt_config_value({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert fmt_config_value(500) == "500"


@pytest.mark.parametrize(
    "value, expected",
    [(None, "n/a"), (0.7999, "below"), (0.80, "within"), (0.85, "within"), (0.8501, "above")],
)
def test_target_status_boundaries_are_inclusive(value, expected):
    assert target_status(value, 0.80, 0.85)[0] == expected


def test_default_output_path():
    assert default_output_path("a", "b").as_posix() == "reports/a__vs__b.html"


# ---------------------------------------------------------------- verdict banner

def test_pass_context(repo, data_files):
    ctx = build(repo, data_files, *same_rows())
    assert (ctx.verdict.label, ctx.verdict.css) == ("PASS", "verdict-pass")
    assert ctx.verdict.reasons
    assert ctx.generated_at == "2026-10-23T10:00:00+00:00"
    assert ctx.title == "base vs cand"
    assert ctx.drilldowns == []                                  # nothing got worse


def test_fail_context_marks_regressed_metric(repo, data_files):
    base, _ = same_rows()
    cand = [qr("cand", f"q{i}", recall_at_k=1.0, mrr=1.0, token_prf1=0.5, faithfulness=1.0) for i in range(5)]
    ctx = build(repo, data_files, base, cand)
    assert (ctx.verdict.label, ctx.verdict.css) == ("FAIL", "verdict-fail")
    row = next(m for m in ctx.metrics if m.name == "token_prf1")
    assert (row.status, row.status_css) == ("regressed", "status-regressed")
    assert (row.delta, row.delta_css) == ("-0.5000", "delta-worse")


def test_inconclusive_context_never_looks_like_pass(repo, data_files):
    base, _ = same_rows()
    cand = [qr("cand", f"q{i}", **full(1.0)) for i in range(2)] + [
        qr("cand", f"q{i}", status="failed", answer="judge crashed on this") for i in range(2, 5)
    ]                                                             # evaluator failures -> validity 2/5
    ctx = build(repo, data_files, base, cand)
    assert ctx.verdict.label == "INCONCLUSIVE"
    assert ctx.verdict.css == "verdict-inconclusive"
    assert "do not read it as a pass" in ctx.verdict.headline
    assert any("validity" in r for r in ctx.verdict.reasons)
    assert ctx.candidate.validity == "2/5 (40%)"
    assert ctx.baseline.validity == "5/5 (100%)"


def test_no_gated_metrics_gives_empty_breakdowns(repo, data_files):
    base = [qr("base", f"q{i}") for i in range(3)]               # judged, but no metric values at all
    cand = [qr("cand", f"q{i}") for i in range(3)]
    ctx = build(repo, data_files, base, cand)
    assert ctx.verdict.label == "INCONCLUSIVE"
    assert ctx.drilldowns == [] and ctx.groups == []
    assert ctx.worst_summary.startswith("no gated metrics")


# ---------------------------------------------------------------- drill-down selection (D9-Q9)

def _ranking_rows():
    base = [qr("base", f"q{i}", **full(1.0)) for i in range(4)]
    cand = [
        qr("cand", "q0", recall_at_k=0.7, mrr=1.0, token_prf1=1.0, faithfulness=1.0),   # 0.3 on 1 metric
        qr("cand", "q1", recall_at_k=1.0, mrr=0.5, token_prf1=1.0, faithfulness=1.0),   # 0.5 -> first
        qr("cand", "q2", recall_at_k=1.0, mrr=1.0, token_prf1=0.7, faithfulness=0.7),   # 0.3 on 2 metrics
        qr("cand", "q3", **full(1.0)),                                                   # unchanged
    ]
    return base, cand


def test_drilldowns_ranked_by_max_worsening_then_metric_count(repo, data_files):
    ctx = build(repo, data_files, *_ranking_rows())
    assert [d.qa_id for d in ctx.drilldowns] == ["q1", "q2", "q0"]
    assert [d.rank for d in ctx.drilldowns] == [1, 2, 3]
    first = ctx.drilldowns[0]
    assert (first.worst_metric, first.worst_worsening, first.n_metrics_worse) == ("mrr", "0.5000", 1)
    assert ctx.drilldowns[1].n_metrics_worse == 2


def test_drilldowns_cut_at_top(repo, data_files):
    ctx = build(repo, data_files, *_ranking_rows(), top=2)
    assert [d.qa_id for d in ctx.drilldowns] == ["q1", "q2"]
    assert ctx.top == 2


# ---------------------------------------------------------------- drill-down content (D9-Q10)

def test_drilldown_content(repo, data_files):
    base = [qr("base", "q0", retrieved=("p1",), **full(1.0))]
    cand = [qr("cand", "q0", retrieved=("p2", "p2", "p1"), answer="cand answer",
               reason="claim 2 unsupported", recall_at_k=1.0, mrr=0.5, token_prf1=1.0, faithfulness=0.5)]
    ctx = build(repo, data_files, base, cand)
    d = ctx.drilldowns[0]
    assert d.question == "Q0?"
    assert d.gold_answers == ["gold 0"]
    assert d.category == "simple"

    c = d.candidate
    assert c.answer == "cand answer"
    assert c.faithfulness_reason == "claim 2 unsupported"
    assert ("mrr", "0.5000") in c.metrics
    assert [p.passage_id for p in c.passages] == ["p2", "p1"]           # deduped, first-seen order
    assert [p.hits for p in c.passages] == [2, 1]
    assert [p.is_gold for p in c.passages] == [False, True]
    assert c.passages[0].text == "Passage two." and c.passages[0].found
    assert c.gold_not_retrieved == []
    assert c.log_link == "logs/cand.jsonl#q0"
    assert d.baseline.log_link == "logs/base.jsonl#q0"


def test_gold_not_retrieved_is_listed(repo, data_files):
    base = [qr("base", "q0", **full(1.0))]
    cand = [qr("cand", "q0", retrieved=("p2",), recall_at_k=0.0, mrr=0.0, token_prf1=1.0, faithfulness=1.0)]
    ctx = build(repo, data_files, base, cand)
    assert ctx.drilldowns[0].candidate.gold_not_retrieved == ["p1"]


# ---------------------------------------------------------------- degradation (D9-Q11)

def test_missing_passages_file_degrades(repo, data_files, tmp_path):
    ctx = build(repo, data_files, *_ranking_rows(), passages_path=tmp_path / "nope.json")
    assert any("passages file not found" in w for w in ctx.data_warnings)
    p = ctx.drilldowns[0].candidate.passages[0]
    assert (p.found, p.text) == (False, "")


def test_corrupt_passages_file_degrades(repo, data_files, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    ctx = build(repo, data_files, *_ranking_rows(), passages_path=bad)
    assert any("passages file unreadable" in w for w in ctx.data_warnings)


def test_unknown_passage_id_is_warned(repo, data_files):
    base = [qr("base", "q0", **full(1.0))]
    cand = [qr("cand", "q0", retrieved=("p9",), recall_at_k=0.0, mrr=0.0, token_prf1=1.0, faithfulness=1.0)]
    ctx = build(repo, data_files, base, cand)
    assert any("p9" in w and "not found" in w for w in ctx.data_warnings)


def test_missing_split_file_degrades(repo, data_files, tmp_path):
    ctx = build(repo, data_files, *_ranking_rows(), splits_dir=tmp_path / "nosplits")
    assert any("split file not found" in w for w in ctx.data_warnings)
    assert ctx.drilldowns[0].gold_answers == []


# ---------------------------------------------------------------- other sections

def test_latency_rows(repo, data_files):
    ctx = build(
        repo, data_files, *same_rows(),
        base_latency={"latency_p50_total": 1000.0, "latency_p95_total": 2000.0, "latency_p50_judge": 800.0},
        cand_latency={"latency_p50_total": 1500.0, "latency_p95_total": 1800.0},
    )
    assert [r.stage for r in ctx.latency] == ["total", "judge"]
    total, judge = ctx.latency
    assert (total.base_p50, total.cand_p50, total.delta_p50) == ("1,000 ms", "1,500 ms", "+500 ms")
    assert total.delta_p50_css == "delta-worse"                     # slower is worse
    assert (total.delta_p95, total.delta_p95_css) == ("-200 ms", "delta-better")
    assert (judge.cand_p50, judge.delta_p50) == (DASH, DASH)


# ---------------------------------------------------------------- targets file (D9-Q17)

def test_targets_from_file(repo, data_files, targets_file):
    ctx = build(repo, data_files, *same_rows(v=0.6), targets_path=targets_file)
    assert [t.label for t in ctx.targets] == ["Recall@5", "MRR", "Answer F1", "Faithfulness"]
    f1 = next(t for t in ctx.targets if t.metric == "token_prf1")
    assert (f1.target, f1.baseline, f1.baseline_status) == ("0.55–0.65", "0.6000", "within")
    recall = next(t for t in ctx.targets if t.metric == "recall_at_k")
    assert (recall.candidate_status, recall.candidate_css) == ("below", "target-below")
    assert ctx.target_note.startswith("Targets: README.md DEC-03 (2026-09-10).")


def test_no_targets_file_omits_section(repo, data_files):
    ctx = build(repo, data_files, *same_rows())
    assert ctx.targets == []
    assert ctx.target_note == NO_TARGETS_NOTE


def test_target_label_defaults_to_metric_name_and_source_to_path(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("targets:\n  mrr: {low: 0.5, high: 0.6}\n", encoding="utf-8")
    spec = load_targets(path)
    assert spec.ranges["mrr"].label == "mrr"
    assert (spec.ranges["mrr"].low, spec.ranges["mrr"].high) == (0.5, 0.6)
    assert spec.source == "" and spec.path == str(path)


def test_target_for_unknown_metric_is_skipped_with_warning(repo, data_files, tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("targets:\n  mrr: {low: 0.5, high: 0.6}\n  ndcg: {low: 0.5, high: 0.6}\n", encoding="utf-8")
    ctx = build(repo, data_files, *same_rows(), targets_path=path)
    assert [t.metric for t in ctx.targets] == ["mrr"]
    assert any("ndcg" in w for w in ctx.data_warnings)
    assert ctx.target_note.startswith(f"Targets: {path}.")             # no source -> path shown


@pytest.mark.parametrize(
    "content, fragment",
    [
        ("{not: [yaml", "unreadable"),
        ("source: x\n", "non-empty 'targets:'"),
        ("targets: {}\n", "non-empty 'targets:'"),
        ("targets:\n  mrr: 0.7\n", "must be a mapping"),
        ("targets:\n  mrr: {low: 0.6}\n", "missing high"),
        ("targets:\n  mrr: {low: high, high: 0.7}\n", "must be a number"),
        ("targets:\n  mrr: {low: true, high: 0.7}\n", "must be a number"),
        ("targets:\n  mrr: {low: 0.8, high: 0.7}\n", "low 0.8 > high 0.7"),
    ],
)
def test_malformed_targets_file_is_invalid_input(tmp_path, content, fragment):
    path = tmp_path / "t.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(TargetsFileError) as exc_info:
        load_targets(path)
    assert fragment in str(exc_info.value)
    assert exc_info.value.exit_code == 8


def test_missing_targets_file_fails_before_db_work(repo, data_files, tmp_path):
    passages, splits = data_files
    with pytest.raises(TargetsFileError, match="not found"):
        generate_report(repo, "no-such-run", "also-missing", passages_path=passages, splits_dir=splits,
                        targets_path=tmp_path / "nope.yaml", gates=GATES)


def test_config_changes_and_comparator_warnings(repo, data_files):
    base, cand = same_rows()
    ctx = build(repo, data_files, base, cand[:4])                    # q4 only in baseline
    change = next(c for c in ctx.config_changes if c.key == "pipeline.params.chunk_size")
    assert (change.before, change.after) == ("500", "1000")
    assert any("question sets differ" in w for w in ctx.warnings)
    assert ctx.baseline.config_hash == "hash-base" and ctx.baseline.split == "val"


def test_config_hash_is_shortened_with_full_hash_kept(repo, data_files):
    full_hash = "5352aebf303eba8b1d265bf99dc6c63844573dc9e0c0a879331bc486d2e84a39"
    base, cand = same_rows()
    add_run(repo, "base", base, config_hash=full_hash)
    add_run(repo, "cand", cand, chunk=1000)
    passages, splits = data_files
    ctx = generate_report(repo, "base", "cand", passages_path=passages, splits_dir=splits, gates=GATES, now=NOW)
    assert (ctx.baseline.config_hash, ctx.baseline.config_hash_short) == (full_hash, "5352aebf303e")
    assert ctx.candidate.config_hash_short == "hash-cand"              # shorter than 12: unchanged


def test_group_tables(repo, data_files):
    base = [qr("base", f"q{i}", category="simple", **full(1.0)) for i in range(4)] + [
        qr("base", "q4", category="comparison", **full(1.0))]
    cand = [qr("cand", f"q{i}", category="simple", **full(1.0)) for i in range(4)] + [
        qr("cand", "q4", category="comparison", recall_at_k=1.0, mrr=0.0, token_prf1=1.0, faithfulness=1.0)]
    ctx = build(repo, data_files, base, cand)
    by_cat = ctx.groups[0]
    assert by_cat.title == "By category"
    assert by_cat.columns == list(M4)
    comparison = next(r for r in by_cat.rows if r.label == "comparison")
    assert comparison.small_n and comparison.n_worse == 1
    assert comparison.cells[1].startswith("-1.000")
    assert ctx.worst_summary.startswith("1 of 5 common rows")


def test_unknown_run_propagates(repo, data_files):
    add_run(repo, "base", same_rows()[0])
    passages, splits = data_files
    with pytest.raises(UnknownRunError):
        generate_report(repo, "base", "nope", passages_path=passages, splits_dir=splits, gates=GATES)


# ---------------------------------------------------------------- rendering (tiny template, not D9-B4's)

def test_render_autoescapes_untrusted_text(repo, data_files, tmp_path):
    base = [qr("base", "q0", **full(1.0))]
    cand = [qr("cand", "q0", answer="<script>alert(1)</script>",
               recall_at_k=1.0, mrr=0.5, token_prf1=1.0, faithfulness=1.0)]
    ctx = build(repo, data_files, base, cand)
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "index.html").write_text("{{ ctx.drilldowns[0].candidate.answer }}", encoding="utf-8")
    html = render_html(ctx, template_dir=tpl)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_fails_loudly_on_unknown_field(repo, data_files, tmp_path):
    ctx = build(repo, data_files, *same_rows())
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "index.html").write_text("{{ ctx.no_such_field }}", encoding="utf-8")
    with pytest.raises(UndefinedError):
        render_html(ctx, template_dir=tpl)


def test_write_report_creates_parent_dirs(repo, data_files, tmp_path):
    ctx = build(repo, data_files, *same_rows())
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "index.html").write_text("<p>{{ ctx.verdict.label }} — {{ ctx.title }}</p>", encoding="utf-8")
    out = write_report(ctx, tmp_path / "reports" / "nested" / "r.html", template_dir=tpl)
    assert out.read_text(encoding="utf-8") == "<p>PASS — base vs cand</p>"