"""
tests/integration/test_api.py

Day 13 (B3): the HTTP API, through FastAPI's TestClient. No server process, no Ollama, no model.

  - /health          real HTTP probes against throw-away local servers (never port 11434)
  - /runs            POST starts a fake executor on the background thread (a real run is 25 minutes)
  - /compare         real comparator + verdict code over a temporary SQLite
  - /reports/{pair}  real report code over the same database
  - JobManager       one run at a time, redacted errors, no half-written reads

Safety (F-D10-33): every database, config and report path is a temp path; ApiSettings is always passed
explicitly, so the real ragregress.db, artifacts/chroma and .env are never touched (create_app is built
with configure=False: no .env, no logging setup). Fake secrets are assembled at run time so
scripts/check_secrets.py never flags this file.

Run (repo root):  python -m pytest tests/integration/test_api.py -v
"""

from __future__ import annotations

import dataclasses
import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from ragregress.api import experiments
from ragregress.api.health import check_health, ollama_base, probe
from ragregress.api.jobs import FAILED, RUNNING, SUCCEEDED, JobBusy, JobExists, JobManager
from ragregress.api.schemas import clean_number
from ragregress.api.settings import ApiSettings
from ragregress.core.errors import ConfigValidationError, RunFailedError
from ragregress.main import create_app
from ragregress.report import TargetsFileError
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

FAKE_KEY = "sk" + "-" + "A1b2C3d4E5f6G7h8I9j0K1l2"          # assembled at run time (see the docstring)

ENV_NAMES = ("OLLAMA_HOST", "JUDGE_BASE_URL", "JUDGE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Nothing from the developer's shell or .env decides what these tests do."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- helpers

def wait_for(predicate, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the condition")


class FakeExecutor:
    """Stands in for ragregress.core.runner.execute: reports progress, optionally waits, optionally fails."""

    def __init__(self, rows: int = 3, hold: bool = False, error: Optional[BaseException] = None):
        self.rows, self.hold, self.error = rows, hold, error
        self.calls: List[tuple] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self, config, run_id, on_row_done):
        self.calls.append((config, run_id))
        try:
            on_row_done(0, self.rows, 0.0)
            self.started.set()
            if self.hold:
                assert self.release.wait(10), "test forgot to release the fake run"
            if self.error is not None:
                raise self.error
            for done in range(1, self.rows + 1):
                on_row_done(done, self.rows, float(done))
        finally:
            self.finished.set()


CONFIG_TEXT = """\
pipeline:
  id: builtin_rag
  params:
    embedding_model: all-MiniLM-L6-v2
    chunk_size: 500
    overlap: 50
    generation_model: qwen2.5:3b-instruct
    seed: 42
top_k: 5
judge_model: qwen2.5:7b-instruct
judge_seed: 42
eval_split: val
is_dev: false
max_workers: 1
db_path: {db}
"""


class Api:
    def __init__(self, tmp_path, executor: FakeExecutor):
        self.tmp = tmp_path
        self.db = tmp_path / "api.db"
        self.configs = tmp_path / "configs"
        self.configs.mkdir()
        self.executor = executor
        self.settings = ApiSettings(db_path=self.db, configs_dir=self.configs,
                                    passages_path=tmp_path / "no_passages.json", splits_dir=tmp_path / "no_splits",
                                    targets_path=None, top=3)
        self.app = create_app(self.settings, executor=executor, configure=False)
        self.client: TestClient = None            # type: ignore[assignment]
        self.repo = Repository(self.db)

    def write_config(self, name: str = "baseline", text: Optional[str] = None, **replace: str) -> None:
        body = text if text is not None else CONFIG_TEXT.format(db=json.dumps(str(self.db)))
        for old, new in replace.items():
            assert old in body, old
            body = body.replace(old, new)
        (self.configs / f"{name}.yaml").write_text(body, encoding="utf-8")


@pytest.fixture()
def api(tmp_path):
    executor = FakeExecutor()
    handle = Api(tmp_path, executor)
    with TestClient(handle.app) as client:
        handle.client = client
        yield handle
    executor.release.set()


def seed_run(repo: Repository, run_id: str, *, values: Optional[Dict[str, float]] = None, is_dev: bool = False,
             cfg: Optional[Dict[str, Any]] = None, n: int = 10, evaluator_failures: int = 0,
             created_at: str = "2026-10-29T00:00:00+00:00", status: Optional[str] = "healthy",
             detail: Optional[str] = None) -> None:
    config: Dict[str, Any] = {"eval_split": "val", "judge_model": "qwen2.5:7b", "top_k": 5,
                              "pipeline": {"id": "builtin_rag", "params": {"chunk_size": 500}}}
    config.update(cfg or {})
    scores = {"recall_at_k": 1.0, "mrr": 0.8, "token_prf1": 0.5, "faithfulness": 0.9}
    scores.update(values or {})
    repo.create_run(Run(run_id=run_id, created_at=created_at, is_dev=is_dev, k=5, max_workers=1, config=config,
                        config_hash=f"hash-{run_id}", run_status=status, run_status_detail=detail))
    rows = []
    for i in range(n):
        if i < n - evaluator_failures:
            rows.append(QueryResult(run_id=run_id, qa_id=f"q{i}", question=f"question {i}?", query_status="judged",
                                    failure_mode="pass", category="simple", generation_answer="an answer",
                                    metric_results={m: {"value": v} for m, v in scores.items()}))
        else:                                   # an answer exists but the judge crashed: unmeasured
            rows.append(QueryResult(run_id=run_id, qa_id=f"q{i}", question=f"question {i}?", query_status="failed",
                                    failure_mode="judge_failed", category="simple", generation_answer="an answer"))
    repo.save_query_results(rows)
    repo.save_metrics(run_id, {**scores, "latency_p50_total": 1200.0})


@pytest.fixture()
def seeded(api):
    seed_run(api.repo, "base")
    seed_run(api.repo, "same")
    return api


# ---------------------------------------------------------------- /health

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append((self.path, self.headers.get("Authorization")))
        self.send_response(self.server.status)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@contextmanager
def fake_server(status: int = 200):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.status, server.requests = status, []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def url_of(server) -> str:
    return f"http://127.0.0.1:{server.server_port}"


def closed_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def test_health_reports_both_reachable_and_uses_the_ollama_path():
    with fake_server() as ollama:
        health = check_health({"OLLAMA_HOST": url_of(ollama)})
    assert (health.status, health.ollama_reachable, health.judge_reachable) == ("ok", True, True)
    assert [path for path, _ in ollama.requests] == ["/api/tags"]          # one probe: the judge shares the endpoint


def test_health_ollama_down_means_judge_down_when_it_shares_the_endpoint():
    health = check_health({"OLLAMA_HOST": closed_url()})
    assert (health.ollama_reachable, health.judge_reachable) == (False, False)


def test_health_judge_base_url_is_probed_on_its_own_models_path():
    with fake_server() as judge:
        health = check_health({"OLLAMA_HOST": closed_url(), "JUDGE_BASE_URL": url_of(judge) + "/v1/"})
    assert (health.ollama_reachable, health.judge_reachable) == (False, True)
    assert [path for path, _ in judge.requests] == ["/v1/models"]


def test_health_judge_down_while_ollama_is_up():
    with fake_server() as ollama:
        health = check_health({"OLLAMA_HOST": url_of(ollama), "JUDGE_BASE_URL": closed_url()})
    assert (health.ollama_reachable, health.judge_reachable) == (True, False)


def test_health_any_http_answer_counts_as_reachable():
    with fake_server(status=401) as server:
        assert probe(url_of(server) + "/models") is True
        assert check_health({"OLLAMA_HOST": url_of(server)}).ollama_reachable is True


def test_health_accepts_host_and_port_without_a_scheme():
    with fake_server() as server:
        assert check_health({"OLLAMA_HOST": f"127.0.0.1:{server.server_port}"}).ollama_reachable is True


def test_health_never_requests_a_non_http_url():
    assert probe("file:///etc/passwd") is False
    assert check_health({"OLLAMA_HOST": "file:///etc/passwd"}).ollama_reachable is False
    assert check_health({"OLLAMA_HOST": closed_url(), "JUDGE_BASE_URL": "ftp://example.invalid"}).judge_reachable is False


def test_health_sends_no_key_and_never_echoes_the_environment():
    env = {"OLLAMA_HOST": closed_url(), "JUDGE_API_KEY": FAKE_KEY, "OPENAI_API_KEY": FAKE_KEY}
    with fake_server() as judge:
        env["JUDGE_BASE_URL"] = url_of(judge)
        body = check_health(env).model_dump_json()
    assert [auth for _, auth in judge.requests] == [None]
    assert FAKE_KEY not in body and "127.0.0.1" not in body


def test_health_default_endpoint_is_localhost_11434():
    assert ollama_base({}) == "http://localhost:11434"                     # the string only: nothing probes it here
    assert ollama_base({"OLLAMA_HOST": "  box:9  "}) == "http://box:9"
    assert ollama_base({"OLLAMA_HOST": "https://h:1/"}) == "https://h:1"


def test_health_endpoint_is_200_even_when_nothing_is_reachable(api, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", closed_url())
    response = api.client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama_reachable": False, "judge_reachable": False}


# ---------------------------------------------------------------- start-up and settings

def test_startup_creates_the_database_so_an_empty_api_lists_no_runs(api):
    assert api.db.is_file()
    assert api.client.get("/runs").json() == []


def test_targets_file_is_used_only_when_it_exists(tmp_path):
    present = tmp_path / "t.yaml"
    present.write_text("targets: {}", encoding="utf-8")
    assert ApiSettings(targets_path=present).targets_file() == present
    assert ApiSettings(targets_path=tmp_path / "missing.yaml").targets_file() is None
    assert ApiSettings(targets_path=None).targets_file() is None


def test_openapi_page_lists_the_documented_endpoints(api):
    paths = api.client.get("/openapi.json").json()["paths"]
    assert {(p, m) for p, ms in paths.items() for m in ms} >= {
        ("/health", "get"), ("/runs", "get"), ("/runs", "post"), ("/runs/{run_id}", "get"),
        ("/compare", "post"), ("/reports/{pair}", "get")}
    assert api.client.get("/docs").status_code == 200


def test_nan_and_infinity_become_null_in_json():
    assert clean_number(float("nan")) is None and clean_number(float("inf")) is None
    assert clean_number(float("-inf")) is None and clean_number(None) is None
    assert clean_number(0.25) == 0.25 and clean_number(0.0) == 0.0


# ---------------------------------------------------------------- GET /runs, GET /runs/{id}

def test_runs_are_listed_newest_first_with_the_dev_flag(api):
    seed_run(api.repo, "old", created_at="2026-10-01T00:00:00+00:00")
    seed_run(api.repo, "new", created_at="2026-10-30T00:00:00+00:00", status=None)
    seed_run(api.repo, "dev1", created_at="2026-10-15T00:00:00+00:00", is_dev=True, cfg={"eval_split": "dev"})
    rows = api.client.get("/runs").json()
    assert [r["run_id"] for r in rows] == ["new", "dev1", "old"]
    assert rows[0] == {"run_id": "new", "created_at": "2026-10-30T00:00:00+00:00", "is_dev": False, "k": 5,
                       "run_status": None, "eval_split": "val", "pipeline_id": "builtin_rag", "config_hash": "hash-new"}
    assert rows[1]["is_dev"] is True and rows[1]["eval_split"] == "dev"


def test_run_list_does_not_carry_error_text(api):
    seed_run(api.repo, "r", status="unhealthy", detail="boom")
    assert "run_status_detail" not in api.client.get("/runs").json()[0]


def test_run_detail_has_stored_metrics_and_config(api):
    seed_run(api.repo, "r")
    body = api.client.get("/runs/r").json()
    assert body["run_id"] == "r" and body["run_status"] == "healthy"
    assert body["metrics"]["recall_at_k"] == 1.0 and body["metrics"]["latency_p50_total"] == 1200.0
    assert body["config"]["judge_model"] == "qwen2.5:7b"
    assert body["job"] is None


def test_run_detail_redacts_the_stored_status_detail(api):
    seed_run(api.repo, "r", status="unhealthy", detail=f"judge rejected api_key={FAKE_KEY} and {FAKE_KEY}")
    detail = api.client.get("/runs/r").json()["run_status_detail"]
    assert FAKE_KEY not in detail and "[REDACTED]" in detail


def test_unknown_run_is_404(api):
    assert api.client.get("/runs/nope").status_code == 404


# ---------------------------------------------------------------- POST /runs

def start(api, name: str = "baseline"):
    return api.client.post("/runs", json={"config": name})


def test_post_runs_returns_202_and_runs_the_config_in_the_background(api):
    api.write_config()
    response = start(api)
    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "running" and body["status_url"] == f"/runs/{body['run_id']}"
    assert body["run_id"].startswith("val-")
    wait_for(lambda: api.client.get(body["status_url"]).json()["job"]["state"] == SUCCEEDED)

    (config, run_id), = api.executor.calls
    assert run_id == body["run_id"]
    assert config.eval_split == "val" and config.judge_model == "qwen2.5:7b-instruct"
    job = api.client.get(body["status_url"]).json()["job"]
    assert (job["rows_done"], job["rows_total"], job["error"]) == (3, 3, None)
    assert job["finished_at"] is not None


def test_post_runs_while_a_run_is_in_progress_is_409_and_progress_is_visible(api):
    api.executor.hold = True
    api.write_config("a")
    api.write_config("b")
    first = start(api, "a").json()
    assert api.executor.started.wait(10)

    detail = api.client.get(f"/runs/{first['run_id']}").json()          # row not stored yet (the fake writes none)
    assert detail["job"]["state"] == RUNNING and detail["job"]["rows_total"] == 3
    assert detail["run_status"] is None and detail["metrics"] == {}
    assert detail["is_dev"] is None and detail["k"] is None                # unknown, not invented

    second = start(api, "b")
    assert second.status_code == 409 and first["run_id"] in second.json()["detail"]
    assert len(api.executor.calls) == 1

    api.executor.release.set()
    wait_for(lambda: api.client.get(f"/runs/{first['run_id']}").json()["job"]["state"] == SUCCEEDED)


def test_a_failed_run_shows_a_redacted_error(api):
    api.executor.error = RunFailedError("rid", f"judge said api_key={FAKE_KEY}")
    api.write_config()
    run_id = start(api).json()["run_id"]
    wait_for(lambda: api.client.get(f"/runs/{run_id}").json()["job"]["state"] == FAILED)
    text = json.dumps(api.client.get(f"/runs/{run_id}").json())
    assert FAKE_KEY not in text and "[REDACTED]" in text


def test_an_unexpected_exception_is_a_failed_job_not_a_crash(api):
    api.executor.error = ValueError("bad thing")
    api.write_config()
    run_id = start(api).json()["run_id"]
    wait_for(lambda: api.client.get(f"/runs/{run_id}").json()["job"]["state"] == FAILED)
    assert api.client.get(f"/runs/{run_id}").json()["job"]["error"] == "unexpected ValueError: bad thing"
    assert api.client.get("/health").status_code == 200                    # the API survived


def test_an_unexpected_exception_message_is_redacted_too(api):
    api.executor.error = ValueError(f"library said api_key={FAKE_KEY}")
    api.write_config()
    run_id = start(api).json()["run_id"]
    wait_for(lambda: api.client.get(f"/runs/{run_id}").json()["job"]["state"] == FAILED)
    error = api.client.get(f"/runs/{run_id}").json()["job"]["error"]
    assert error.startswith("unexpected ValueError") and FAKE_KEY not in error and "[REDACTED]" in error


def test_unknown_config_name_is_404_and_nothing_runs(api):
    assert start(api, "nope").status_code == 404
    assert api.executor.calls == []


@pytest.mark.parametrize("name", ["../secret", "a/b", "..", ".hidden", "x y", "", "a" * 65])
def test_config_names_cannot_escape_the_configs_folder(api, name):
    (api.tmp / "secret.yaml").write_text(CONFIG_TEXT.format(db=json.dumps(str(api.db))), encoding="utf-8")
    assert start(api, name).status_code == 422
    assert api.executor.calls == []


def test_an_invalid_config_is_422_with_the_validation_message(api):
    api.write_config(**{"top_k: 5": "top_k: five"})
    response = start(api)
    assert response.status_code == 422 and "top_k" in response.json()["detail"]
    assert api.executor.calls == []


def test_a_config_for_another_database_is_422(api):
    api.write_config(**{json.dumps(str(api.db)): json.dumps(str(api.tmp / "other.db"))})
    response = start(api)
    assert response.status_code == 422 and "db_path" in response.json()["detail"]
    assert api.executor.calls == []


def test_a_judge_that_cannot_be_set_up_is_422_before_the_run_starts(api):
    api.write_config(**{"judge_model: qwen2.5:7b-instruct": "judge_provider: openai\njudge_model: gpt-4o-mini"})
    response = start(api)
    assert response.status_code == 422 and "API key" in response.json()["detail"]
    assert api.executor.calls == []


def test_a_prefixed_judge_model_is_accepted_on_a_non_openai_url_and_the_key_stays_out_of_the_reply(api, monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", FAKE_KEY)
    monkeypatch.setenv("JUDGE_BASE_URL", "http://127.0.0.1:9")
    api.write_config(**{"judge_model: qwen2.5:7b-instruct": "judge_provider: openai\njudge_model: openai/gpt-4o"})
    response = start(api)                                                  # prefix rule B allows a non-OpenAI URL
    assert response.status_code == 202
    assert FAKE_KEY not in response.text


def test_extra_fields_in_the_run_request_are_rejected(api):
    api.write_config()
    assert api.client.post("/runs", json={"config": "baseline", "db_path": "x.db"}).status_code == 422
    assert api.client.post("/runs", json={}).status_code == 422


def test_a_run_id_that_is_already_stored_is_409(api, monkeypatch):
    api.write_config()
    seed_run(api.repo, "val-fixed-id")
    monkeypatch.setattr(experiments, "default_run_id", lambda cfg: "val-fixed-id")
    response = start(api)
    assert response.status_code == 409 and "val-fixed-id" in response.json()["detail"]
    assert api.executor.calls == []


# ---------------------------------------------------------------- POST /compare

def compare(api, baseline: str, candidates: List[str]):
    return api.client.post("/compare", json={"baseline": baseline, "candidates": candidates})


def metric(entry: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(m for m in entry["comparison"]["metrics"] if m["name"] == name)


def test_compare_one_candidate_gives_the_cli_verdict_as_json(seeded):
    response = compare(seeded, "base", ["same"])
    assert response.status_code == 200
    body = response.json()
    assert body["baseline"] == "base" and [e["candidate"] for e in body["results"]] == ["same"]
    entry = body["results"][0]
    assert (entry["ok"], entry["error"]) == (True, None)
    cmp = entry["comparison"]
    assert (cmp["verdict"], cmp["exit_code"]) == ("PASS", 0)
    assert cmp["gated"] == ["recall_at_k", "mrr", "token_prf1", "faithfulness"]
    assert cmp["validity_baseline"] == {"measured": 10, "attempted": 10, "ratio": 1.0}
    assert cmp["baseline"]["run_id"] == "base" and cmp["candidate"]["run_id"] == "same"
    assert cmp["baseline"]["config_hash"] == "hash-base" and cmp["baseline"]["legacy"] is False
    assert cmp["config_changes"] == [] and cmp["warnings"] == []
    recall = metric(entry, "recall_at_k")
    assert recall == {"name": "recall_at_k", "status": "ok", "before": 1.0, "after": 1.0, "delta": 0.0,
                      "n_common": 10, "better": 0, "worse": 0, "tie": 10, "threshold": 0.05,
                      "higher_is_better": True, "note": ""}


def test_compare_many_candidates_keeps_their_order_and_each_verdict(seeded):
    seed_run(seeded.repo, "worse", values={"recall_at_k": 0.0})
    seed_run(seeded.repo, "unsure", evaluator_failures=5)
    body = compare(seeded, "base", ["worse", "same", "unsure"]).json()
    assert [e["candidate"] for e in body["results"]] == ["worse", "same", "unsure"]
    verdicts = [(e["comparison"]["verdict"], e["comparison"]["exit_code"]) for e in body["results"]]
    assert verdicts == [("FAIL", 6), ("PASS", 0), ("INCONCLUSIVE", 7)]
    assert metric(body["results"][0], "recall_at_k")["status"] == "regressed"
    assert metric(body["results"][0], "recall_at_k")["delta"] == -1.0
    assert "validity" in " ".join(body["results"][2]["comparison"]["reasons"])


def test_fail_and_inconclusive_are_http_200(seeded):
    seed_run(seeded.repo, "worse", values={"recall_at_k": 0.0})
    assert compare(seeded, "base", ["worse"]).status_code == 200


def test_one_bad_candidate_does_not_stop_the_others(seeded):
    seed_run(seeded.repo, "devrun", is_dev=True, cfg={"eval_split": "dev"})
    seed_run(seeded.repo, "other_split", cfg={"eval_split": "test"})
    body = compare(seeded, "base", ["ghost", "same", "devrun", "other_split"]).json()
    ok = {e["candidate"]: e["ok"] for e in body["results"]}
    assert ok == {"ghost": False, "same": True, "devrun": False, "other_split": False}
    errors = {e["candidate"]: e["error"] for e in body["results"] if not e["ok"]}
    assert "ghost" in errors["ghost"]
    assert "dev" in errors["devrun"].lower()
    assert "eval_split" in errors["other_split"]
    assert all(e["comparison"] is None for e in body["results"] if not e["ok"])
    assert not any(e.startswith("unexpected") for e in errors.values())      # input problems are not bugs


def test_unknown_baseline_is_404(seeded):
    assert compare(seeded, "ghost", ["same"]).status_code == 404


def test_a_dev_baseline_makes_every_entry_an_error(seeded):
    seed_run(seeded.repo, "devrun", is_dev=True, cfg={"eval_split": "dev"})
    body = compare(seeded, "devrun", ["same"]).json()
    assert body["results"][0]["ok"] is False and "dev" in body["results"][0]["error"].lower()


def test_a_judge_mismatch_shows_as_an_ungated_metric_with_its_reason(seeded):
    seed_run(seeded.repo, "other_judge", cfg={"judge_model": "llama3:8b"})
    entry = compare(seeded, "base", ["other_judge"]).json()["results"][0]
    faith = metric(entry, "faithfulness")
    assert faith["status"] == "ungated" and "judge_model differs" in faith["note"]
    assert "faithfulness" not in entry["comparison"]["gated"]
    assert any("judge_model differs" in w for w in entry["comparison"]["warnings"])
    assert faith["delta"] == 0.0                                           # still computed, just not gated


def test_config_changes_are_listed_with_missing_keys_flagged(seeded):
    seed_run(seeded.repo, "bigger", cfg={"pipeline": {"id": "builtin_rag", "params": {"chunk_size": 1000, "extra": 1}}})
    changes = {c["key"]: c for c in compare(seeded, "base", ["bigger"]).json()["results"][0]["comparison"]["config_changes"]}
    assert changes["pipeline.params.chunk_size"] == {"key": "pipeline.params.chunk_size", "before": 500, "after": 1000,
                                                     "before_missing": False, "after_missing": False}
    assert changes["pipeline.params.extra"]["before_missing"] is True
    seed_run(seeded.repo, "plainer", cfg={"pipeline": {"id": "builtin_rag", "params": {}}})
    back = {c["key"]: c for c in compare(seeded, "bigger", ["plainer"]).json()["results"][0]["comparison"]["config_changes"]}
    assert back["pipeline.params.extra"]["after_missing"] is True and back["pipeline.params.extra"]["before_missing"] is False
    assert back["pipeline.params.extra"]["after"] is None and back["pipeline.params.extra"]["before"] == 1
    assert changes["pipeline.params.extra"]["before"] is None and changes["pipeline.params.extra"]["after"] == 1


@pytest.mark.parametrize("payload", [
    {"baseline": "base", "candidates": []},
    {"baseline": "base", "candidates": ["same", "same"]},
    {"baseline": "base", "candidates": [f"c{i}" for i in range(11)]},
    {"baseline": "base", "candidates": ["bad/id"]},
    {"baseline": "../base", "candidates": ["same"]},
    {"baseline": "base", "candidate": "same"},
    {"baseline": "base", "candidates": ["same"], "extra": 1},
    {"candidates": ["same"]},
])
def test_bad_compare_requests_are_422(seeded, payload):
    assert seeded.client.post("/compare", json=payload).status_code == 422


def test_ten_candidates_is_allowed(seeded):
    for i in range(10):
        seed_run(seeded.repo, f"c{i}")
    assert len(compare(seeded, "base", [f"c{i}" for i in range(10)]).json()["results"]) == 10


def test_a_crash_inside_one_comparison_is_that_entrys_error(seeded, monkeypatch):
    real = experiments.compare_runs

    def flaky(repo, baseline, candidate):
        if candidate == "same":
            raise RuntimeError(f"boom api_key={FAKE_KEY}")
        return real(repo, baseline, candidate)

    seed_run(seeded.repo, "fine")
    monkeypatch.setattr(experiments, "compare_runs", flaky)
    body = compare(seeded, "base", ["same", "fine"]).json()
    assert [e["ok"] for e in body["results"]] == [False, True]
    assert body["results"][0]["error"].startswith("unexpected RuntimeError") and FAKE_KEY not in body["results"][0]["error"]


# ---------------------------------------------------------------- GET /reports/{pair}

def test_report_is_html_with_the_verdict_banner(seeded):
    response = seeded.client.get("/reports/base__vs__same")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert 'class="banner verdict-pass"' in response.text
    assert "<title>PASS · base vs same</title>" in response.text


def test_report_for_a_failing_pair_says_fail(seeded):
    seed_run(seeded.repo, "worse", values={"recall_at_k": 0.0})
    assert 'class="banner verdict-fail"' in seeded.client.get("/reports/base__vs__worse").text


def test_the_html_file_name_the_cli_writes_works_too(seeded):
    assert seeded.client.get("/reports/base__vs__same.html").status_code == 200


def test_report_for_an_unknown_run_is_404(seeded):
    assert seeded.client.get("/reports/base__vs__ghost").status_code == 404
    assert seeded.client.get("/reports/ghost__vs__same").status_code == 404


@pytest.mark.parametrize("pair", ["base", "base__vs__", "__vs__same", "base__vs__a__vs__b", "ba d__vs__same",
                                  "..__vs__same", "base__vs__.same"])
def test_a_malformed_report_id_is_422(seeded, pair):
    assert seeded.client.get(f"/reports/{pair}").status_code == 422


def test_a_slash_in_the_report_id_never_reaches_the_report_code(seeded):
    assert seeded.client.get("/reports/base__vs__sa/me").status_code == 404


def test_report_for_runs_that_cannot_be_compared_is_400(seeded):
    seed_run(seeded.repo, "devrun", is_dev=True, cfg={"eval_split": "dev"})
    response = seeded.client.get("/reports/base__vs__devrun")
    assert response.status_code == 400 and "dev" in response.json()["detail"].lower()


def test_a_broken_targets_file_is_a_server_error_with_a_redacted_message(seeded, monkeypatch):
    def broken(*args, **kwargs):
        raise TargetsFileError("configs/targets/x.yaml", f"bad token={FAKE_KEY}")

    monkeypatch.setattr(experiments, "generate_report", broken)
    response = seeded.client.get("/reports/base__vs__same")
    assert response.status_code == 500 and FAKE_KEY not in response.text


def test_a_missing_default_targets_file_means_no_target_table_not_an_error(api, tmp_path):
    seed_run(api.repo, "base")
    seed_run(api.repo, "same")
    gone = create_app(dataclasses.replace(api.settings, targets_path=tmp_path / "gone.yaml"),
                      executor=api.executor, configure=False)
    with TestClient(gone) as client:
        assert client.get("/reports/base__vs__same").status_code == 200


def test_report_gets_the_servers_paths_and_targets_only_when_present(api, tmp_path, monkeypatch):
    seed_run(api.repo, "base")
    seed_run(api.repo, "same")
    seen: List[Dict[str, Any]] = []
    real = experiments.generate_report

    def spy(repo, baseline, candidate, **kwargs):
        seen.append(kwargs)
        return real(repo, baseline, candidate, **kwargs)

    monkeypatch.setattr(experiments, "generate_report", spy)
    api.client.get("/reports/base__vs__same")
    assert seen[0]["targets_path"] is None and seen[0]["top"] == 3
    assert seen[0]["passages_path"] == api.settings.passages_path and seen[0]["splits_dir"] == api.settings.splits_dir

    targets = tmp_path / "targets.yaml"
    targets.write_text("targets: {}", encoding="utf-8")
    second = create_app(dataclasses.replace(api.settings, targets_path=targets), executor=api.executor, configure=False)
    with TestClient(second) as client:
        client.get("/reports/base__vs__same")
    assert seen[1]["targets_path"] == targets


# ---------------------------------------------------------------- JobManager

def make_config():
    from ragregress.core.config import load_run_config
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "c.yaml"
        path.write_text(CONFIG_TEXT.format(db=json.dumps(str(pathlib.Path(folder) / "x.db"))), encoding="utf-8")
        return load_run_config(path)


def test_job_manager_allows_one_run_at_a_time_then_the_next():
    executor = FakeExecutor(hold=True)
    jobs = JobManager(executor)
    config = make_config()
    jobs.start(config, "r1", "c")
    assert executor.started.wait(10)
    with pytest.raises(JobBusy) as busy:
        jobs.start(config, "r2", "c")
    assert busy.value.run_id == "r1"
    executor.release.set()
    wait_for(lambda: jobs.get("r1").state == SUCCEEDED)
    jobs.start(config, "r2", "c")                                          # free again
    wait_for(lambda: jobs.get("r2").state == SUCCEEDED)


def test_job_manager_refuses_a_run_id_it_already_used():
    jobs = JobManager(FakeExecutor())
    config = make_config()
    jobs.start(config, "r1", "c")
    wait_for(lambda: jobs.get("r1").state == SUCCEEDED)
    with pytest.raises(JobExists):
        jobs.start(config, "r1", "c")
    assert jobs.knows("r1") and not jobs.knows("r9") and jobs.get("r9") is None


def test_job_manager_hands_out_copies():
    executor = FakeExecutor(hold=True)
    jobs = JobManager(executor)
    jobs.start(make_config(), "r1", "c")
    assert executor.started.wait(10)
    copy = jobs.get("r1")
    copy.state = "tampered"
    assert jobs.get("r1").state == RUNNING
    executor.release.set()
    wait_for(lambda: jobs.get("r1").state == SUCCEEDED)


def test_job_manager_records_progress_and_a_failure_with_a_redacted_message():
    executor = FakeExecutor(rows=5, error=ConfigValidationError(f"token={FAKE_KEY}"))
    jobs = JobManager(executor)
    jobs.start(make_config(), "r1", "c")
    wait_for(lambda: jobs.get("r1").state == FAILED)
    job = jobs.get("r1")
    assert (job.rows_done, job.rows_total) == (0, 5)
    assert FAKE_KEY not in job.error and "[REDACTED]" in job.error
    assert job.finished_at is not None


def test_a_failed_run_frees_the_slot():
    jobs = JobManager(FakeExecutor(error=ValueError("x")))
    config = make_config()
    jobs.start(config, "r1", "c")
    wait_for(lambda: jobs.get("r1").state == FAILED)
    jobs.start(config, "r2", "c")
    wait_for(lambda: jobs.get("r2").state == FAILED)