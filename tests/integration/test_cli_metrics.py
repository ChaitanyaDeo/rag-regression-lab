"""CLI metric selection: `ragregress metrics` and `run --metric` (dry run, no models needed)."""
import json

import pytest
from typer.testing import CliRunner

from ragregress.cli import app
from ragregress.evaluation.metrics import METRIC_REGISTRY
from ragregress.evaluation.metrics.selection import default_metric_names

runner = CliRunner()

CONFIG = """\
pipeline:
  id: builtin_rag
  params: {{embedding_model: all-MiniLM-L6-v2, chunk_size: 500, overlap: 50, generation_model: "qwen2.5:3b-instruct", seed: 42}}
top_k: 5
judge_provider: openai
judge_model: gpt-4o-mini
judge_seed: 42
eval_split: val
is_dev: false
max_workers: 1
db_path: {db}
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)     # openai judge with no key
    monkeypatch.chdir(tmp_path)                            # no stray .env is picked up
    path = tmp_path / "run.yaml"
    path.write_text(CONFIG.format(db=tmp_path / "r.db"))
    return str(path)


def test_metrics_command_lists_every_registered_metric_and_the_default_set():
    out = runner.invoke(app, ["metrics"])
    assert out.exit_code == 0
    for name in METRIC_REGISTRY:
        assert name in out.output
    assert f"default: {', '.join(default_metric_names())}" in out.output


def test_metrics_json_is_the_catalog():
    out = runner.invoke(app, ["metrics", "--json"])
    assert out.exit_code == 0
    assert [row["name"] for row in json.loads(out.output)] == list(METRIC_REGISTRY)


def test_dry_run_without_judged_metric_needs_no_judge_key(cfg):
    out = runner.invoke(app, ["run", "--config", cfg, "--dry-run", "--metric", "rouge_l,bleu"])
    assert out.exit_code == 0, out.output
    assert "metrics      : rouge_l, bleu" in out.output
    assert "not needed" in out.output


def test_dry_run_default_selection_still_checks_the_judge(cfg):
    out = runner.invoke(app, ["run", "--config", cfg, "--dry-run"])
    assert out.exit_code == 3


def test_unknown_metric_exits_3_and_names_it(cfg):
    out = runner.invoke(app, ["run", "--config", cfg, "--dry-run", "--metric", "nope"])
    assert out.exit_code == 3
    assert "nope" in out.output


def test_metric_all_selects_every_registered_metric(cfg, monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    out = runner.invoke(app, ["run", "--config", cfg, "--dry-run", "--metric", "all"])
    assert out.exit_code == 0, out.output
    assert f"metrics      : {', '.join(METRIC_REGISTRY)}" in out.output


def test_run_passes_the_selected_names_to_execute(cfg, monkeypatch):
    from ragregress import cli
    seen = {}

    def fake_execute(cfg_, run_id=None, reindex=False, on_row_done=None, metrics=None):
        seen["metrics"] = metrics
        raise cli.RagRegressError("stop here")          # no outcome needed; exit code 1

    monkeypatch.setattr(cli, "execute", fake_execute)
    runner.invoke(app, ["run", "--config", cfg, "--metric", "bleu,exact_match"])
    assert seen["metrics"] == ["bleu", "exact_match"]

def test_metrics_table_marks_the_metrics_that_use_the_judge():
    rows = {line.split()[0]: line.split()[2] for line in runner.invoke(app, ["metrics"]).output.splitlines()[1:]
            if line.strip() and not line.startswith(("default:", "select"))}
    assert rows["faithfulness"] == rows["context_recall"] == rows["context_precision"] == "yes"
    assert rows["bleu"] == rows["recall_at_k"] == "-"