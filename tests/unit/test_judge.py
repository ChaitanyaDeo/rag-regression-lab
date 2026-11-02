"""
tests/unit/test_judge.py

Day 13 (B2c): the judge is configurable. Two run-config fields say WHAT judges:
    judge_provider: ollama | openai     (default ollama; left out of the dump while it is the default)
    judge_model:    the model name only
Endpoint and key come from the environment (JUDGE_BASE_URL, JUDGE_API_KEY, OLLAMA_HOST, ...).

  A  the two fields           provider is a closed set; the model is a plain name, never "provider/model"
  B  resolve_judge            env -> endpoint / key; ollama needs no key; openai needs one; display is safe
  C  RunConfig                validated at load (exit 3); old configs keep their hashes
  D  build_judge / runner     the right arguments reach the client; execute() fails fast on a missing key
  E  CLI --dry-run            prints provider / model / endpoint, never the key; .env is honoured
  F  on the wire              the REAL instructor client against a local fake server: URL, key header, model name

Safety: nothing here can reach a real service. Section F binds 127.0.0.1 on a random port and always sets
JUDGE_BASE_URL / OLLAMA_HOST to it; the fixture clears every judge-related variable first. Fake keys are
assembled at runtime so scripts/check_secrets.py never flags this file.

Run (repo root):  python -m pytest tests/unit/test_judge.py -v
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel, ValidationError

from ragregress.core import runner
from ragregress.core.config import RunConfig, load_run_config
from ragregress.core.errors import ConfigValidationError
from ragregress.core.logging import APP_LOGGER
from ragregress.evaluation.judge import (
    API_KEY_ENV, BASE_URL_ENV, OLLAMA_HOST_ENV, OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV, PLACEHOLDER_KEY, PROVIDERS,
    build_judge, check_judge_model, check_judge_provider, resolve_judge,
)

JUDGE_ENV = (BASE_URL_ENV, API_KEY_ENV, OLLAMA_HOST_ENV, OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV, "LOG_LEVEL")
KEY = "sk" + "-judgetestkey1234567890abcd"          # fake, assembled at runtime
OTHER_KEY = "sk" + "-otherjudgekey0987654321zyxw"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in JUDGE_ENV:
        monkeypatch.setenv(name, "unset")           # record the original value so teardown restores it,
        monkeypatch.delenv(name)                    # even when the CLI's load_dotenv writes os.environ
    monkeypatch.chdir(tmp_path)                     # the CLI loads .env from the cwd: never the repo's own
    root, app = logging.getLogger(), logging.getLogger(APP_LOGGER)
    saved = (list(root.handlers), root.level, app.level)
    yield
    for h in list(root.handlers):
        if h not in saved[0]:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved[1])
    app.setLevel(saved[2])


# ================================================================ A. the two fields

def test_supported_providers():
    assert PROVIDERS == ("ollama", "openai")


@pytest.mark.parametrize("value, expected", [("ollama", "ollama"), ("openai", "openai"), ("  OpenAI ", "openai"), ("OLLAMA", "ollama")])
def test_provider_is_canonicalised(value, expected):
    assert check_judge_provider(value) == expected


@pytest.mark.parametrize("value", ["anthropic", "google", "Groq", "deepseek", "", "openai/gpt-4o", "azure_openai"])
def test_unsupported_provider_is_rejected_with_the_way_out(value):
    with pytest.raises(ValueError) as err:
        check_judge_provider(value)
    message = str(err.value)
    assert "unsupported judge_provider" in message and "ollama, openai" in message and BASE_URL_ENV in message


@pytest.mark.parametrize("value", [
    "qwen2.5:7b-instruct", "gpt-4o-mini", "hf.co/org/model:q4_K_M", "org/model-v2", "llama3:8b", "  qwen2.5:7b  ",
    "anthropic/claude-sonnet",                     # not one of OUR providers: an ordinary slash in a model id
])
def test_plain_model_names_are_accepted(value):
    check_judge_model(value)            # no exception


def test_check_judge_model_reads_no_environment_so_prefixes_are_not_its_business():
    check_judge_model("openai/gpt-4o")          # the prefix rule needs the endpoint: see resolve_judge


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_model_is_rejected(value):
    with pytest.raises(ValueError):
        check_judge_model(value)


# ================================================================ B. environment -> connection

def test_ollama_defaults_to_localhost():
    r = resolve_judge("ollama", "qwen2.5:7b-instruct", env={})
    assert (r.provider, r.model, r.base_url, r.api_key) == ("ollama", "qwen2.5:7b-instruct", "http://localhost:11434/v1", None)
    assert r.endpoint == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    "host, expected",
    [
        ("http://host.docker.internal:11434", "http://host.docker.internal:11434/v1"),
        ("http://gpu-box:8080/", "http://gpu-box:8080/v1"),
        ("localhost:11434", "http://localhost:11434/v1"),                           # Ollama's own host:port form
        ("gpu-box", "http://gpu-box:11434/v1"),                                    # default port added
        ("https://ollama.example.org", "https://ollama.example.org/v1"),            # https: no port invented
        ("  http://localhost:11434  ", "http://localhost:11434/v1"),
        ("", "http://localhost:11434/v1"),                                          # blank = unset
    ],
)
def test_ollama_host_becomes_the_v1_endpoint(host, expected):
    assert resolve_judge("ollama", "qwen", env={OLLAMA_HOST_ENV: host}).base_url == expected


def test_ollama_base_url_override_is_used_verbatim_and_wins_over_ollama_host():
    env = {BASE_URL_ENV: "http://127.0.0.1:9999/v1", OLLAMA_HOST_ENV: "http://elsewhere:11434"}
    assert resolve_judge("ollama", "qwen", env=env).base_url == "http://127.0.0.1:9999/v1"


@pytest.mark.parametrize("host", ["http://host:notaport", "http://", "http://host:99999"])
def test_unusable_ollama_host_is_a_config_error(host):
    with pytest.raises(ConfigValidationError) as err:
        resolve_judge("ollama", "qwen", env={OLLAMA_HOST_ENV: host})
    assert err.value.exit_code == 3 and OLLAMA_HOST_ENV in str(err.value)


def test_ollama_sends_no_key_and_says_so_when_one_is_set():
    r = resolve_judge("ollama", "qwen", env={API_KEY_ENV: KEY})
    assert r.api_key is None
    assert "ignored" in r.key_note and "judge_provider: openai" in r.key_note
    assert KEY not in r.key_note


def test_openai_with_a_judge_key():
    r = resolve_judge("openai", "gpt-4o-mini", env={API_KEY_ENV: KEY})
    assert (r.provider, r.model, r.api_key, r.base_url) == ("openai", "gpt-4o-mini", KEY, None)
    assert r.endpoint == "https://api.openai.com/v1"
    assert r.key_note == f"set (from {API_KEY_ENV})"


def test_openai_falls_back_to_the_openai_key_variable_and_judge_key_wins():
    only_openai = resolve_judge("openai", "gpt-4o-mini", env={OPENAI_API_KEY_ENV: OTHER_KEY})
    assert only_openai.api_key == OTHER_KEY and OPENAI_API_KEY_ENV in only_openai.key_note
    both = resolve_judge("openai", "gpt-4o-mini", env={API_KEY_ENV: KEY, OPENAI_API_KEY_ENV: OTHER_KEY})
    assert both.api_key == KEY


def test_openai_without_any_key_or_base_url_is_a_config_error_exit_3():
    with pytest.raises(ConfigValidationError) as err:
        resolve_judge("openai", "gpt-4o-mini", env={})
    assert err.value.exit_code == 3
    assert API_KEY_ENV in str(err.value) and OPENAI_API_KEY_ENV in str(err.value)


def test_openai_blank_key_counts_as_missing():
    with pytest.raises(ConfigValidationError):
        resolve_judge("openai", "gpt-4o-mini", env={API_KEY_ENV: "   ", OPENAI_API_KEY_ENV: ""})


def test_openai_compatible_server_without_a_key_gets_a_placeholder():
    r = resolve_judge("openai", "my-model", env={BASE_URL_ENV: "http://127.0.0.1:8000/v1"})
    assert r.api_key == PLACEHOLDER_KEY and r.base_url == "http://127.0.0.1:8000/v1"
    assert "placeholder" in r.key_note


def test_openai_base_url_variable_also_counts_and_is_left_to_the_sdk():
    r = resolve_judge("openai", "my-model", env={OPENAI_BASE_URL_ENV: "http://127.0.0.1:8000/v1"})
    assert r.api_key == PLACEHOLDER_KEY
    assert r.base_url is None                                     # the SDK reads OPENAI_BASE_URL itself
    assert r.endpoint == "http://127.0.0.1:8000/v1"               # ... but the plan shows where it will go


def test_display_endpoint_has_no_credentials_or_query_but_the_client_gets_the_real_url():
    url = f"https://alice:{KEY}@judge.example.org/v1?token={OTHER_KEY}"
    r = resolve_judge("openai", "m", env={BASE_URL_ENV: url, API_KEY_ENV: KEY})
    assert r.endpoint == "https://judge.example.org/v1"
    assert r.base_url == url


def test_the_key_never_appears_in_repr_or_notes():
    r = resolve_judge("openai", "gpt-4o-mini", env={API_KEY_ENV: KEY})
    assert KEY not in repr(r) and KEY not in r.key_note and KEY not in r.endpoint
    assert r.api_key == KEY                                       # ... but the client still gets it


def test_a_bad_provider_or_model_is_a_config_error_here_too():
    with pytest.raises(ConfigValidationError) as err:
        resolve_judge("anthropic", "claude", env={})
    assert err.value.exit_code == 3 and "judge_provider" in str(err.value)


OPENROUTER = "https://openrouter.ai/api/v1"


@pytest.mark.parametrize("model, prefix", [
    ("openai/gpt-4o-mini", "openai"), ("ollama/qwen2.5:7b-instruct", "ollama"), ("OpenAI/gpt-4o", "openai"),
    ("  openai /gpt-4o", "openai"),
])
@pytest.mark.parametrize("env", [{}, {BASE_URL_ENV: "http://127.0.0.1:11434/v1"}])
def test_a_provider_prefix_is_rejected_under_ollama_whatever_the_endpoint(model, prefix, env):
    with pytest.raises(ConfigValidationError) as err:
        resolve_judge("ollama", model, env=env)
    assert err.value.exit_code == 3 and err.value.source == "judge_model"
    assert "separate fields" in str(err.value)
    if prefix == "ollama":                                          # already the chosen provider
        assert "keep only the model name" in str(err.value) and "set judge_provider" not in str(err.value)
    else:
        assert f"set judge_provider: {prefix}" in str(err.value)


@pytest.mark.parametrize("env", [
    {API_KEY_ENV: KEY},                                             # no base URL at all = api.openai.com
    {API_KEY_ENV: KEY, BASE_URL_ENV: "https://api.openai.com/v1"},  # explicitly OpenAI
    {API_KEY_ENV: KEY, OPENAI_BASE_URL_ENV: "https://API.OpenAI.com/v1"},
])
def test_a_provider_prefix_is_rejected_on_real_openai(env):
    with pytest.raises(ConfigValidationError) as err:
        resolve_judge("openai", "openai/gpt-4o-mini", env=env)
    assert err.value.source == "judge_model"
    assert "keep only the model name" in str(err.value) and "set judge_provider" not in str(err.value)


def test_a_prefix_naming_the_other_provider_says_which_provider_to_set():
    with pytest.raises(ConfigValidationError) as err:
        resolve_judge("openai", "ollama/qwen2.5:7b", env={API_KEY_ENV: KEY})
    assert "set judge_provider: ollama" in str(err.value)


@pytest.mark.parametrize("env", [
    {API_KEY_ENV: KEY, BASE_URL_ENV: OPENROUTER},
    {API_KEY_ENV: KEY, OPENAI_BASE_URL_ENV: OPENROUTER},            # the SDK's own variable counts too
    {BASE_URL_ENV: "http://127.0.0.1:8000/v1"},                     # a local OpenAI-compatible server
])
@pytest.mark.parametrize("model", ["openai/gpt-4o", "openai/gpt-4o-mini", "OpenAI/gpt-4o", "ollama/qwen2.5:7b"])
def test_a_prefixed_id_is_accepted_through_a_non_openai_endpoint(model, env):
    r = resolve_judge("openai", model, env=env)
    assert (r.provider, r.model) == ("openai", model.strip())       # passed through untouched


def test_a_judge_base_url_that_is_not_openai_beats_an_openai_base_url_that_is():
    env = {API_KEY_ENV: KEY, BASE_URL_ENV: OPENROUTER, OPENAI_BASE_URL_ENV: "https://api.openai.com/v1"}
    assert resolve_judge("openai", "openai/gpt-4o", env=env).model == "openai/gpt-4o"
    env = {API_KEY_ENV: KEY, BASE_URL_ENV: "https://api.openai.com/v1", OPENAI_BASE_URL_ENV: OPENROUTER}
    with pytest.raises(ConfigValidationError):                      # the URL actually used is JUDGE_BASE_URL
        resolve_judge("openai", "openai/gpt-4o", env=env)


@pytest.mark.parametrize("model", ["hf.co/org/model:q4_K_M", "org/model-v2", "anthropic/claude-sonnet", "google/gemini"])
@pytest.mark.parametrize("provider, env", [("ollama", {}), ("openai", {API_KEY_ENV: KEY}), ("openai", {BASE_URL_ENV: OPENROUTER})])
def test_other_slashes_are_always_fine(model, provider, env):
    assert resolve_judge(provider, model, env=env).model == model


def test_provider_is_canonicalised_and_model_is_trimmed_when_resolving():
    r = resolve_judge(" OpenAI ", "  gpt-4o-mini ", env={API_KEY_ENV: KEY})
    assert (r.provider, r.model) == ("openai", "gpt-4o-mini")


def test_default_env_is_os_environ(monkeypatch):
    monkeypatch.setenv(OLLAMA_HOST_ENV, "http://from-os-environ:11434")
    assert resolve_judge("ollama", "qwen").base_url == "http://from-os-environ:11434/v1"


def test_resolve_does_not_modify_the_env_mapping():
    env = {API_KEY_ENV: KEY}
    resolve_judge("openai", "m", env=env)
    assert env == {API_KEY_ENV: KEY}


# ================================================================ C. RunConfig

def raw_config(judge_model="qwen2.5:7b-instruct", judge_provider=None):
    raw = {
        "pipeline": {"id": "builtin_rag", "params": {
            "embedding_model": "all-MiniLM-L6-v2", "chunk_size": 500, "overlap": 50,
            "generation_model": "qwen2.5:3b-instruct"}},
        "top_k": 5, "judge_model": judge_model, "eval_split": "val", "is_dev": False,
    }
    if judge_provider is not None:
        raw["judge_provider"] = judge_provider
    return raw


def test_judge_provider_defaults_to_ollama():
    assert RunConfig.model_validate(raw_config()).judge_provider == "ollama"


def test_the_default_provider_is_left_out_of_the_dump_so_old_hashes_are_unchanged():
    cfg = RunConfig.model_validate(raw_config())
    assert "judge_provider" not in cfg.model_dump(mode="json")
    assert set(cfg.hash_payload()) == {"pipeline", "top_k", "judge_model", "judge_seed", "eval_split", "is_dev"}


def test_a_non_default_provider_is_dumped_and_hashed():
    cfg = RunConfig.model_validate(raw_config("gpt-4o-mini", "openai"))
    assert cfg.model_dump(mode="json")["judge_provider"] == "openai"
    assert cfg.hash_payload()["judge_provider"] == "openai"


def test_an_explicit_default_provider_hashes_the_same_as_leaving_it_out():
    omitted = RunConfig.model_validate(raw_config()).config_hash()
    assert RunConfig.model_validate(raw_config(judge_provider="ollama")).config_hash() == omitted
    assert RunConfig.model_validate(raw_config(judge_provider="  OLLAMA ")).config_hash() == omitted


def test_the_provider_is_canonicalised_at_load():
    assert RunConfig.model_validate(raw_config("gpt-4o", " OpenAI ")).judge_provider == "openai"


def test_a_different_provider_or_model_is_a_different_run():
    h = lambda model, provider=None: RunConfig.model_validate(raw_config(model, provider)).config_hash()
    assert h("gpt-4o-mini") != h("gpt-4o-mini", "openai")          # same model name, other service
    assert h("qwen2.5:7b-instruct") != h("llama3:8b")


@pytest.mark.parametrize("provider", ["anthropic", "google", "", 5])
def test_unsupported_provider_fails_validation_at_the_judge_provider_field(provider):
    with pytest.raises(ValidationError) as err:
        RunConfig.model_validate(raw_config(judge_provider=provider))
    assert [e["loc"] for e in err.value.errors()] == [("judge_provider",)]


def test_unsupported_provider_error_says_how_to_reach_another_service():
    with pytest.raises(ValidationError) as err:
        RunConfig.model_validate(raw_config(judge_provider="anthropic"))
    assert BASE_URL_ENV in err.value.errors()[0]["msg"]


@pytest.mark.parametrize("model", ["   ", ""])
def test_empty_judge_model_fails_validation_at_the_judge_model_field(model):
    with pytest.raises(ValidationError) as err:
        RunConfig.model_validate(raw_config(model))
    assert [e["loc"] for e in err.value.errors()] == [("judge_model",)]


@pytest.mark.parametrize("model, provider", [("openai/gpt-4o", "openai"), ("openai/gpt-4o-mini", None), ("ollama/qwen", "openai")])
def test_a_prefixed_model_loads_and_is_stored_as_written_the_prefix_rule_runs_at_resolve_time(model, provider):
    cfg = RunConfig.model_validate(raw_config(model, provider))      # no environment is read at load
    assert cfg.judge_model == model


def test_a_bad_provider_is_reported_at_judge_provider_only():
    with pytest.raises(ValidationError) as err:
        RunConfig.model_validate(raw_config("openai/gpt-4o", "anthropic"))
    assert [e["loc"] for e in err.value.errors()] == [("judge_provider",)]


def test_load_run_config_maps_a_bad_provider_to_exit_3(tmp_path):
    import yaml
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw_config("claude-sonnet", "anthropic")), encoding="utf-8")
    with pytest.raises(ConfigValidationError) as err:
        load_run_config(path)
    assert err.value.exit_code == 3
    assert [loc for loc, _ in err.value.errors] == ["judge_provider"]
    assert "unsupported judge_provider" in err.value.errors[0][1]


def test_load_run_config_reads_both_fields(tmp_path):
    import yaml
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump(raw_config("gpt-4o-mini", "openai")), encoding="utf-8")
    cfg = load_run_config(path)
    assert (cfg.judge_provider, cfg.judge_model) == ("openai", "gpt-4o-mini")


def test_environment_does_not_change_the_hash(monkeypatch):
    before = RunConfig.model_validate(raw_config("gpt-4o-mini", "openai")).config_hash()
    monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:1/v1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    assert RunConfig.model_validate(raw_config("gpt-4o-mini", "openai")).config_hash() == before


# ================================================================ D. build_judge / runner

class FakeJudge:
    """Stands in for InstructorStructuredLLM; records every constructor argument."""
    instances: list = []

    def __init__(self, model_name, provider="ollama", max_retries=2, seed=None, base_url=None, api_key=None):
        self.args = dict(model_name=model_name, provider=provider, max_retries=max_retries, seed=seed,
                         base_url=base_url, api_key=api_key)
        FakeJudge.instances.append(self)


@pytest.fixture
def fake_judge(monkeypatch):
    import ragregress.evaluation.structured_llm as sl
    FakeJudge.instances = []
    monkeypatch.setattr(sl, "InstructorStructuredLLM", FakeJudge)
    return FakeJudge


def test_instructor_client_gets_base_url_and_api_key_only_when_they_are_set(monkeypatch):
    import sys
    import types
    from ragregress.evaluation.structured_llm import InstructorStructuredLLM

    calls = []
    fake = types.SimpleNamespace(
        Mode=types.SimpleNamespace(JSON="json"),
        from_provider=lambda name, mode, **kwargs: calls.append((name, mode, kwargs)) or object(),
    )
    monkeypatch.setitem(sys.modules, "instructor", fake)

    InstructorStructuredLLM("qwen2.5:7b-instruct")                                  # exactly as before B2c
    InstructorStructuredLLM("gpt-4o-mini", provider="openai", base_url="http://x/v1", api_key=KEY)
    InstructorStructuredLLM("m", base_url="http://y/v1")                            # a url without a key

    assert calls == [
        ("ollama/qwen2.5:7b-instruct", "json", {}),
        ("openai/gpt-4o-mini", "json", {"base_url": "http://x/v1", "api_key": KEY}),
        ("ollama/m", "json", {"base_url": "http://y/v1"}),
    ]


def test_build_judge_ollama_arguments(fake_judge):
    build_judge("ollama", "qwen2.5:7b-instruct", seed=7, env={OLLAMA_HOST_ENV: "http://box:11434"})
    assert fake_judge.instances[-1].args == dict(
        model_name="qwen2.5:7b-instruct", provider="ollama", max_retries=3, seed=7,
        base_url="http://box:11434/v1", api_key=None)


def test_build_judge_openai_arguments(fake_judge):
    build_judge("openai", "gpt-4o-mini", seed=5, max_retries=1, env={API_KEY_ENV: KEY})
    assert fake_judge.instances[-1].args == dict(
        model_name="gpt-4o-mini", provider="openai", max_retries=1, seed=5, base_url=None, api_key=KEY)


def test_build_metrics_builds_the_judge_from_the_config_and_the_environment(fake_judge, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:8000/v1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    cfg = RunConfig.model_validate({**raw_config("gpt-4o-mini", "openai"), "judge_seed": 5})

    metrics = runner.build_metrics(cfg)

    assert fake_judge.instances[-1].args == dict(
        model_name="gpt-4o-mini", provider="openai", max_retries=3, seed=5,
        base_url="http://127.0.0.1:8000/v1", api_key=KEY)
    faithfulness = next(m for m in metrics if m.name == "faithfulness")
    assert faithfulness.llm is fake_judge.instances[-1]


def test_build_metrics_with_the_default_provider_uses_ollama(fake_judge):
    runner.build_metrics(RunConfig.model_validate(raw_config()))
    args = fake_judge.instances[-1].args
    assert (args["provider"], args["model_name"], args["api_key"]) == ("ollama", "qwen2.5:7b-instruct", None)


def test_execute_fails_fast_on_a_missing_judge_key_before_building_anything(tmp_path, monkeypatch):
    cfg = RunConfig.model_validate({**raw_config("gpt-4o-mini", "openai"), "db_path": str(tmp_path / "r.db")})
    monkeypatch.setattr(runner, "load_qa_pairs", lambda path: [])

    def must_not_run(*args, **kwargs):
        raise AssertionError("models were built before the judge settings were checked")

    monkeypatch.setattr(runner, "build_rag_system", must_not_run)
    monkeypatch.setattr(runner, "build_metrics", must_not_run)
    with pytest.raises(ConfigValidationError) as err:
        runner.execute(cfg, run_id="val-x-1")
    assert err.value.exit_code == 3 and API_KEY_ENV in str(err.value)


# ================================================================ E. CLI --dry-run

def write_config(tmp_path, judge_model, judge_provider=None):
    import yaml
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({**raw_config(judge_model, judge_provider), "db_path": str(tmp_path / "r.db")}),
                    encoding="utf-8")
    return path


def dry_run(tmp_path, judge_model, judge_provider=None):
    from typer.testing import CliRunner
    import ragregress.cli as cli
    config = write_config(tmp_path, judge_model, judge_provider)
    return CliRunner().invoke(cli.app, ["run", "--config", str(config), "--dry-run"])


def test_dry_run_shows_the_default_ollama_judge(tmp_path):
    result = dry_run(tmp_path, "qwen2.5:7b-instruct")
    assert result.exit_code == 0, result.output
    assert "judge        : ollama / qwen2.5:7b-instruct" in result.output
    assert "judge url    : http://localhost:11434/v1" in result.output
    assert "judge key    : not used" in result.output


def test_dry_run_openai_shows_endpoint_and_key_source_but_never_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:8000/v1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    result = dry_run(tmp_path, "gpt-4o-mini", "openai")
    assert result.exit_code == 0, result.output
    assert "judge        : openai / gpt-4o-mini" in result.output
    assert "judge url    : http://127.0.0.1:8000/v1" in result.output
    assert f"judge key    : set (from {API_KEY_ENV})" in result.output
    assert KEY not in result.output


def test_dry_run_without_the_openai_key_exits_3_like_the_real_run_would(tmp_path):
    result = dry_run(tmp_path, "gpt-4o-mini", "openai")
    assert result.exit_code == 3
    assert API_KEY_ENV in result.output and "Traceback" not in result.output


def test_dry_run_with_an_unsupported_provider_exits_3(tmp_path):
    result = dry_run(tmp_path, "claude-sonnet", "anthropic")
    assert result.exit_code == 3
    assert "judge_provider" in result.output and "unsupported" in result.output


def test_dry_run_with_a_provider_prefix_under_ollama_exits_3(tmp_path):
    result = dry_run(tmp_path, "openai/gpt-4o-mini")                # provider omitted = ollama
    assert result.exit_code == 3
    assert "judge_model" in result.output and "set judge_provider: openai" in result.output


def test_dry_run_with_a_provider_prefix_on_real_openai_exits_3(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    result = dry_run(tmp_path, "openai/gpt-4o-mini", "openai")
    assert result.exit_code == 3
    assert "keep only the model name" in result.output and KEY not in result.output


def test_dry_run_accepts_an_openrouter_style_model_id_through_a_non_openai_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, OPENROUTER)
    monkeypatch.setenv(API_KEY_ENV, KEY)
    result = dry_run(tmp_path, "openai/gpt-4o", "openai")
    assert result.exit_code == 0, result.output
    assert "judge        : openai / openai/gpt-4o" in result.output
    assert f"judge url    : {OPENROUTER}" in result.output
    assert KEY not in result.output


def test_judge_settings_are_read_from_dotenv_in_the_working_directory(tmp_path):
    (tmp_path / ".env").write_text(f"JUDGE_BASE_URL=http://127.0.0.1:8123/v1\nJUDGE_API_KEY={KEY}\n", encoding="utf-8")
    result = dry_run(tmp_path, "gpt-4o-mini", "openai")
    assert result.exit_code == 0, result.output
    assert "judge url    : http://127.0.0.1:8123/v1" in result.output
    assert KEY not in result.output


def test_a_shell_variable_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("JUDGE_BASE_URL=http://127.0.0.1:8123/v1\n", encoding="utf-8")
    monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:8456/v1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    assert "judge url    : http://127.0.0.1:8456/v1" in dry_run(tmp_path, "gpt-4o-mini", "openai").output


def test_ollama_host_from_dotenv_reaches_the_judge_endpoint(tmp_path):
    (tmp_path / ".env").write_text("OLLAMA_HOST=http://host.docker.internal:11434\n", encoding="utf-8")
    assert "judge url    : http://host.docker.internal:11434/v1" in dry_run(tmp_path, "qwen2.5:7b-instruct").output


# ================================================================ F. on the wire (real instructor, fake server)

class Reply(BaseModel):
    answer: str


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        data = json.dumps({
            "id": "fake", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps({"answer": "ok"})}}],
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
def server():
    pytest.importorskip("instructor")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.seen = []
    assert srv.server_port != 11434                                # never the real Ollama
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def ask(judge):
    assert judge.generate_structured("hello", Reply).answer == "ok"


def test_wire_ollama_judge_follows_ollama_host(server, monkeypatch):
    monkeypatch.setenv(OLLAMA_HOST_ENV, f"http://127.0.0.1:{server.server_port}")
    ask(build_judge("ollama", "qwen2.5:7b-instruct", seed=3, max_retries=0))
    (seen,) = server.seen
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "qwen2.5:7b-instruct"          # the model name as given, nothing added
    assert seen["body"]["seed"] == 3 and seen["body"]["temperature"] == 0


def test_wire_judge_base_url_beats_ollama_host(server, monkeypatch):
    monkeypatch.setenv(OLLAMA_HOST_ENV, "http://127.0.0.1:1")                     # nothing listens there
    monkeypatch.setenv(BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1")
    ask(build_judge("ollama", "qwen2.5:7b-instruct", seed=3, max_retries=0))
    assert len(server.seen) == 1


def test_wire_openai_judge_sends_the_key_and_the_model_name(server, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    ask(build_judge("openai", "gpt-4o-mini", seed=9, max_retries=0))
    (seen,) = server.seen
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert seen["body"]["seed"] == 9


def test_wire_openai_judge_uses_the_openai_key_variable_as_a_fallback(server, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv(OPENAI_API_KEY_ENV, OTHER_KEY)
    ask(build_judge("openai", "gpt-4o-mini", seed=9, max_retries=0))
    assert server.seen[0]["auth"] == f"Bearer {OTHER_KEY}"


def test_wire_keyless_server_gets_the_placeholder(server, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1")
    ask(build_judge("openai", "local-model", seed=None, max_retries=0))
    assert server.seen[0]["auth"] == f"Bearer {PLACEHOLDER_KEY}"
    assert "seed" not in server.seen[0]["body"]