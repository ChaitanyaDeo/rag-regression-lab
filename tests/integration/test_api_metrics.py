"""API metric selection: GET /metrics and POST /runs {"metrics": [...]} (fake executor, no models)."""
import threading

import pytest
from fastapi.testclient import TestClient

from ragregress.api.settings import ApiSettings
from ragregress.evaluation.metrics import METRIC_REGISTRY
from ragregress.main import create_app

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


class Recorder:
    """Executor that records how it was called instead of running anything."""

    def __init__(self):
        self.calls = []
        self.called = threading.Event()

    def __call__(self, config, run_id, on_row_done, **kwargs):
        self.calls.append(kwargs)
        self.called.set()


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def client(tmp_path, monkeypatch, recorder):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)          # the openai judge has no key
    configs = tmp_path / "configs"
    configs.mkdir()
    db = tmp_path / "api.db"
    (configs / "demo.yaml").write_text(CONFIG.format(db=db))
    settings = ApiSettings(db_path=db, configs_dir=configs)
    with TestClient(create_app(settings, executor=recorder, configure=False)) as c:
        yield c


def test_metrics_endpoint_lists_the_registry_in_order(client):
    body = client.get("/metrics").json()
    assert [m["name"] for m in body] == list(METRIC_REGISTRY)
    assert {m["name"] for m in body if m["default_enabled"]} == {
        n for n, cls in METRIC_REGISTRY.items() if cls.default_enabled}


def test_metrics_endpoint_says_which_metrics_need_a_judge(client):
    by_name = {m["name"]: m for m in client.get("/metrics").json()}
    assert by_name["faithfulness"]["requires_judge"] is True
    assert by_name["bleu"]["requires_judge"] is False


def test_selected_metrics_reach_the_executor(client, recorder):
    r = client.post("/runs", json={"config": "demo", "metrics": ["bleu", "exact_match"]})
    assert r.status_code == 202, r.text
    assert recorder.called.wait(5)
    assert recorder.calls == [{"metrics": ["bleu", "exact_match"]}]


def test_omitted_metrics_call_the_executor_as_before(client, monkeypatch, recorder):
    monkeypatch.setenv("JUDGE_API_KEY", "k")                    # defaults include the judged metric
    r = client.post("/runs", json={"config": "demo"})
    assert r.status_code == 202, r.text
    assert recorder.called.wait(5)
    assert recorder.calls == [{}]


def test_answer_only_selection_needs_no_judge_key(client):
    assert client.post("/runs", json={"config": "demo", "metrics": ["rouge_l"]}).status_code == 202


def test_default_selection_still_needs_the_judge_key(client, recorder):
    r = client.post("/runs", json={"config": "demo"})
    assert r.status_code == 422
    assert not recorder.calls


def test_judged_selection_needs_the_judge_key(client):
    assert client.post("/runs", json={"config": "demo", "metrics": ["faithfulness"]}).status_code == 422


def test_unknown_metric_is_422_and_nothing_starts(client, recorder):
    r = client.post("/runs", json={"config": "demo", "metrics": ["bleu", "nope"]})
    assert r.status_code == 422
    assert "nope" in r.text
    assert not recorder.called.is_set()


@pytest.mark.parametrize("metrics", [[], ["bleu", "bleu"], ["Bad Name"], "bleu", ["x"] * 33])
def test_malformed_metrics_field_is_422(client, metrics):
    assert client.post("/runs", json={"config": "demo", "metrics": metrics}).status_code == 422