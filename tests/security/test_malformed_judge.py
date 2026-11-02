"""
tests/security/test_malformed_judge.py

what happens when the LLM judge answers with garbage.

  L1  real Faithfulness + a fake judge        -> judge errors propagate (no partial score);
                                                 out-of-range verdicts must not inflate the score
  L2  real InstructorStructuredLLM + real instructor, talking to a tiny local fake
      Ollama (OpenAI-compatible) server       -> garbage raises after max_retries + 1 requests,
                                                 a transient bad reply recovers, seed / temperature
                                                 reach the wire

The pipeline turns a raised judge error into `judge_failed` (test_pipeline_judge.py,
test_pipeline_failures.py). The NaN-row behaviour is B5-L3 (separate file).

Known defects are `xfail(strict=True)` tests that assert the DESIRED behaviour: 
    when the schema is tightened (verdict: Literal[0, 1]) they XPASS, strict mode
    fails the suite, and the marker must be removed.

Safety: the fake server binds 127.0.0.1 on a random free port. `instructor.from_provider`
is patched so the client can never reach the real Ollama (localhost:11434).
No Ollama, no models, no DB, no files.

Run (repo root):  python -m pytest tests/security/test_malformed_judge.py -v
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

instructor = pytest.importorskip("instructor")

from ragregress.evaluation.metrics import Faithfulness
from ragregress.evaluation.metrics.faithfulness.utils import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)
from ragregress.evaluation.structured_llm import InstructorStructuredLLM
from ragregress.rag.interface import Chunk

CONTEXT = [Chunk(chunk_id="p1::0", source_passage_id="p1", text="Newton formulated the law of gravitation.")]
GOOD_CLAIMS = json.dumps({"statements": ["Newton formulated the law of gravitation."]})


def verdict_json(*verdicts):
    return json.dumps({"statements": [
        {"statement": f"s{i}", "reason": "r", "verdict": v} for i, v in enumerate(verdicts)
    ]})


# ================================================================ L1: Faithfulness + fake judge

class ScriptedJudge:
    """Stands in for a StructuredLLM. `claims` / `verdicts` are either a value or an Exception to raise."""

    def __init__(self, claims=("s1",), verdicts=((("s1"), 1),)):
        self.claims, self.verdicts, self.calls = claims, verdicts, []

    def generate_structured(self, prompt, schema):
        self.calls.append(schema)
        payload = self.claims if schema is StatementGeneratorOutput else self.verdicts
        if isinstance(payload, Exception):
            raise payload
        if schema is StatementGeneratorOutput:
            return StatementGeneratorOutput(statements=list(payload))
        return NLIStatementOutput(statements=[
            StatementFaithfulnessAnswer(statement=s, reason="r", verdict=v) for s, v in payload
        ])


def faith(judge):
    return Faithfulness(llm=judge).score(question="q?", answer="an answer", context=CONTEXT)


def test_judge_failure_while_extracting_claims_propagates():
    judge = ScriptedJudge(claims=RuntimeError("claims step exploded"))
    with pytest.raises(RuntimeError, match="claims step exploded"):
        faith(judge)
    assert judge.calls == [StatementGeneratorOutput]          # the NLI step never ran


def test_judge_failure_while_judging_claims_propagates():
    judge = ScriptedJudge(verdicts=RuntimeError("nli step exploded"))
    with pytest.raises(RuntimeError, match="nli step exploded"):
        faith(judge)
    assert judge.calls == [StatementGeneratorOutput, NLIStatementOutput]


def test_more_verdicts_than_statements_is_nan_with_reason():
    r = faith(ScriptedJudge(claims=["s1"], verdicts=[("s1", 1), ("s2", 1), ("s3", 1)]))
    assert math.isnan(r.value)
    assert r.reason == "judge returned 3 verdicts for 1 statements"


@pytest.mark.parametrize("bad", [2, -1, 7])
@pytest.mark.xfail(strict=True, reason="known: verdict is an unconstrained int and bool(verdict) counts any "
                                       "non-zero as supported (F-D11-13); fix post-Day 14: Literal[0, 1]")
def test_out_of_range_verdict_does_not_count_as_supported(bad):
    try:
        r = faith(ScriptedJudge(claims=["s1"], verdicts=[("s1", bad)]))
    except Exception:
        return                                                 # rejecting it is also acceptable
    assert r.value != 1.0


# ================================================================ L2: real instructor + fake Ollama

class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server = self.server
        server.requests.append(body)
        index = min(len(server.requests) - 1, len(server.script) - 1)      # last reply repeats
        data = json.dumps({
            "id": "fake", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": server.script[index]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama(monkeypatch):
    """serve(script) -> server. script: replies by request number; the last one repeats."""
    servers = []
    real_from_provider = instructor.from_provider

    def serve(script):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.script, server.requests = list(script), []
        assert server.server_port != 11434                     # never the real Ollama
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        servers.append(server)
        return server

    def current():
        return servers[-1]

    def patched(spec, **kwargs):
        assert "base_url" not in kwargs
        return real_from_provider(spec, base_url=f"http://127.0.0.1:{current().server_port}/v1", **kwargs)

    monkeypatch.setattr(instructor, "from_provider", patched)
    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


def judge_llm(max_retries=2, seed=42):
    return InstructorStructuredLLM("qwen2.5:7b-instruct", max_retries=max_retries, seed=seed)


@pytest.mark.parametrize("max_retries", [0, 1, 3])             # 3 = what runner.build_metrics uses
def test_garbage_reply_raises_after_max_retries_plus_one_requests(fake_ollama, max_retries):
    server = fake_ollama(["this is not json"])
    llm = judge_llm(max_retries=max_retries)                   # built after serve(): patched client needs the port
    with pytest.raises(Exception):
        llm.generate_structured("claims?", StatementGeneratorOutput)
    assert len(server.requests) == max_retries + 1


def test_one_bad_reply_then_a_good_one_recovers(fake_ollama):
    server = fake_ollama(["garbage", GOOD_CLAIMS])
    result = judge_llm(max_retries=2).generate_structured("claims?", StatementGeneratorOutput)
    assert isinstance(result, StatementGeneratorOutput)
    assert result.statements == ["Newton formulated the law of gravitation."]
    assert len(server.requests) == 2


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"statements": []}),                        # NLI min_length=1
        json.dumps({"statements": [{"statement": "s", "reason": "r", "verdict": "yes"}]}),   # not an int
        json.dumps({"statements": [{"statement": "s", "reason": "r"}]}),                      # verdict missing
        json.dumps({"wrong": "shape"}),
    ],
    ids=["empty-list", "non-int-verdict", "missing-verdict", "wrong-shape"],
)
def test_schema_invalid_nli_reply_raises(fake_ollama, reply):
    server = fake_ollama([reply])
    with pytest.raises(Exception):
        judge_llm(max_retries=2).generate_structured("judge?", NLIStatementOutput)
    assert len(server.requests) == 3


@pytest.mark.xfail(strict=True, reason="known: schema accepts verdict outside {0, 1} (F-D11-13); "
                                       "fix post-Day 14: Literal[0, 1]")
@pytest.mark.parametrize("bad", [2, -1])
def test_out_of_range_verdict_is_rejected_by_the_schema(fake_ollama, bad):
    fake_ollama([verdict_json(bad)])
    with pytest.raises(Exception):
        judge_llm(max_retries=0).generate_structured("judge?", NLIStatementOutput)


def test_real_faithfulness_raises_when_claims_reply_is_garbage(fake_ollama):
    server = fake_ollama(["garbage"])
    with pytest.raises(Exception):
        Faithfulness(llm=judge_llm(max_retries=1)).score(question="q?", answer="an answer", context=CONTEXT)
    assert len(server.requests) == 2                           # claims step only, 1 + 1 retry


def test_real_faithfulness_raises_when_nli_reply_is_garbage(fake_ollama):
    server = fake_ollama([GOOD_CLAIMS, "garbage"])             # claims OK, then garbage forever
    with pytest.raises(Exception):
        Faithfulness(llm=judge_llm(max_retries=1)).score(question="q?", answer="an answer", context=CONTEXT)
    assert len(server.requests) == 1 + 2                       # 1 claims request + NLI (1 + 1 retry)


def test_real_faithfulness_scores_a_well_formed_reply(fake_ollama):
    fake_ollama([GOOD_CLAIMS, verdict_json(1)])
    r = Faithfulness(llm=judge_llm()).score(question="q?", answer="an answer", context=CONTEXT)
    assert r.value == pytest.approx(1.0)


def test_seed_and_temperature_reach_the_wire(fake_ollama):
    server = fake_ollama([GOOD_CLAIMS])
    judge_llm(seed=1234).generate_structured("claims?", StatementGeneratorOutput)
    body = server.requests[0]
    assert body["seed"] == 1234 and body["temperature"] == 0


def test_no_seed_means_no_seed_on_the_wire(fake_ollama):
    server = fake_ollama([GOOD_CLAIMS])
    judge_llm(seed=None).generate_structured("claims?", StatementGeneratorOutput)
    assert "seed" not in server.requests[0] and server.requests[0]["temperature"] == 0