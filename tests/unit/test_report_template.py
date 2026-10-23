"""
tests/unit/test_report_template.py

Renders the REAL ragregress/templates/index.html from contexts built over a temporary SQLite
database. Checks that every verdict renders without errors (StrictUndefined would raise on a
misspelled field), that the banner matches the verdict, that untrusted text is escaped and that
the page is self-contained.

Run (repo root):  python -m pytest tests/unit/test_report_template.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html.parser import HTMLParser

import pytest

from ragregress.evaluation.regression import Gate
from ragregress.report import NO_TARGETS_NOTE, generate_report, render_html
from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

M4 = ("recall_at_k", "mrr", "token_prf1", "faithfulness")
GATES = {name: Gate(0.05, True) for name in M4}
NOW = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)

TARGETS_YAML = """\
source: README.md DEC-03 (2026-09-10)
targets:
  recall_at_k:  {label: "Recall@5", low: 0.80, high: 0.85}
  token_prf1:   {label: "Answer F1", low: 0.55, high: 0.65}
"""

SECTION_HEADINGS = (
    "<h2>Runs</h2>",
    "<h2>Metrics (common rows)</h2>",
    "<h2>Target vs actual</h2>",
    "<h2>Latency (informational, not gated)</h2>",
    "<h2>Where regressions are</h2>",
    "<h2>Worst rows (top 5)</h2>",
)


# ---------------------------------------------------------------- builders

@pytest.fixture()
def env(tmp_path):
    db = tmp_path / "test_report_template.db"
    init_db(db)
    passages = tmp_path / "passages.json"
    passages.write_text(json.dumps([
        {"passage_id": "p1", "title": "T1", "text": "Passage one <i>text</i>."},
        {"passage_id": "p2", "title": "T2", "text": "Passage two."},
    ]), encoding="utf-8")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "val.json").write_text(
        json.dumps([{"qa_id": f"q{i}", "answer": [f"gold {i}"]} for i in range(10)]), encoding="utf-8")
    targets = tmp_path / "targets.yaml"
    targets.write_text(TARGETS_YAML, encoding="utf-8")
    return {"repo": Repository(db), "passages": passages, "splits": splits, "targets": targets, "tmp": tmp_path}


def qr(run_id, qa_id, *, status="judged", answer="an answer", retrieved=("p1",), **values):
    return QueryResult(
        run_id=run_id, qa_id=qa_id, question=f"Question {qa_id}?", query_status=status, failure_mode="pass",
        last_error=None, retrieved_ids=list(retrieved), gold_ids=["p1"], generation_answer=answer,
        metric_results={n: {"value": v, "reason": "r" if n == "faithfulness" else None} for n, v in values.items()},
        latency={}, category="simple",
    )


def full(v=1.0):
    return {n: v for n in M4}


def render(env, base_rows, cand_rows, *, targets=True, passages=None, latency=True):
    repo = env["repo"]
    for run_id, rows, chunk in (("base", base_rows, 500), ("cand", cand_rows, 1000)):
        repo.create_run(Run(run_id=run_id, created_at="2026-10-23T00:00:00", is_dev=False, k=5, max_workers=1,
                            config={"pipeline": {"name": "builtin_rag", "params": {"chunk_size": chunk}},
                                    "eval_split": "val", "judge_model": "j"},
                            config_hash=f"hash-{run_id}"))
        repo.save_query_results(rows)
        if latency:
            repo.save_metrics(run_id, {"latency_p50_total": 1000.0 if run_id == "base" else 1200.0,
                                       "latency_p95_total": 2000.0})
    ctx = generate_report(repo, "base", "cand", passages_path=passages or env["passages"],
                          splits_dir=env["splits"], targets_path=env["targets"] if targets else None,
                          gates=GATES, now=NOW)
    return ctx, render_html(ctx)


class _TagBalance(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


# ---------------------------------------------------------------- verdict banner per verdict

def test_pass_renders_all_sections(env):
    ctx, html = render(env, [qr("base", f"q{i}", **full()) for i in range(5)],
                       [qr("cand", f"q{i}", **full()) for i in range(5)])
    assert 'class="banner verdict-pass"' in html
    assert 'banner verdict-fail"' not in html and 'banner verdict-inconclusive"' not in html
    for heading in SECTION_HEADINGS:
        assert heading in html
    assert "No row got worse on a gated metric." in html
    assert "Generated 2026-10-23T10:00:00+00:00" in html


def test_fail_renders_red_banner_and_regressed_status(env):
    base = [qr("base", f"q{i}", **full()) for i in range(5)]
    cand = [qr("cand", f"q{i}", recall_at_k=1.0, mrr=1.0, token_prf1=0.5, faithfulness=1.0) for i in range(5)]
    _, html = render(env, base, cand)
    assert 'class="banner verdict-fail"' in html
    assert 'banner verdict-pass"' not in html
    assert '<td class="status-regressed">regressed</td>' in html
    assert "#1 · " in html                                            # drill-downs rendered


def test_inconclusive_renders_full_warning_banner_never_pass(env):
    base = [qr("base", f"q{i}", **full()) for i in range(5)]
    cand = [qr("cand", f"q{i}", **full()) for i in range(2)] + [
        qr("cand", f"q{i}", status="failed", answer="judge crashed") for i in range(2, 5)]
    ctx, html = render(env, base, cand)
    assert ctx.verdict.label == "INCONCLUSIVE"
    assert 'class="banner verdict-inconclusive"' in html
    assert "<h1>INCONCLUSIVE</h1>" in html
    assert "do not read it as a pass" in html
    assert 'banner verdict-pass"' not in html and "<h1>PASS</h1>" not in html
    assert "candidate 2/5 (40%)" in html


# ---------------------------------------------------------------- content

def test_drilldown_escapes_untrusted_text_and_shows_passages(env):
    base = [qr("base", "q0", **full())]
    cand = [qr("cand", "q0", answer='<script>alert("x")</script>', retrieved=("p2", "p2", "p1"),
               recall_at_k=1.0, mrr=0.5, token_prf1=1.0, faithfulness=1.0)]
    _, html = render(env, base, cand)
    assert "<script" not in html
    assert "&lt;script&gt;" in html
    assert "Passage one &lt;i&gt;text&lt;/i&gt;." in html            # passage text escaped too
    assert "Question q0?" in html and "gold 0" in html
    assert '<span class="gold"> · gold</span>' in html
    assert "2 chunks" in html
    assert 'href="logs/cand.jsonl#q0"' in html and "available Day 13" in html


def test_targets_table_or_note(env):
    rows = ([qr("base", f"q{i}", **full()) for i in range(3)], [qr("cand", f"q{i}", **full()) for i in range(3)])
    _, with_targets = render(env, *rows)
    assert "<td>Recall@5</td>" in with_targets
    assert "Targets: README.md DEC-03 (2026-09-10)." in with_targets


def test_no_targets_shows_note_only(env):
    rows = ([qr("base", f"q{i}", **full()) for i in range(3)], [qr("cand", f"q{i}", **full()) for i in range(3)])
    _, html = render(env, *rows, targets=False)
    assert NO_TARGETS_NOTE in html
    assert "<td>Recall@5</td>" not in html


def test_missing_passages_and_latency_degrade(env):
    base = [qr("base", "q0", **full())]
    cand = [qr("cand", "q0", recall_at_k=1.0, mrr=0.5, token_prf1=1.0, faithfulness=1.0)]
    _, html = render(env, base, cand, passages=env["tmp"] / "nope.json", latency=False)
    assert "Report warnings" in html and "passages file not found" in html
    assert "passage text unavailable" in html
    assert "No latency recorded for these runs." in html


# ---------------------------------------------------------------- page structure

def test_page_is_balanced_and_self_contained(env):
    base = [qr("base", f"q{i}", **full()) for i in range(4)]
    cand = [qr("cand", "q0", recall_at_k=0.0, mrr=0.0, token_prf1=0.2, faithfulness=0.3)] + [
        qr("cand", f"q{i}", **full()) for i in range(1, 4)]
    _, html = render(env, base, cand)
    checker = _TagBalance()
    checker.feed(html)
    assert checker.errors == [] and checker.stack == []
    assert "<script" not in html and "<link" not in html and "src=" not in html
    assert "http://" not in html and "https://" not in html