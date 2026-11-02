"""
tests/unit/test_docker_files.py

Day 13 B2b + B4: static guards for Dockerfile, .dockerignore and docker-compose.yml (the CLI service and the API service).
They read the files as text / YAML; no Docker daemon and no image build are needed.
(One test runs `docker compose config` when the docker CLI is installed; that also needs no daemon.)

What they protect:
  - secrets never reach an image or a compose file (.env, keys, tokens)
  - the build context is an allow-list, so .env / the database / artifacts cannot leak into it
  - the image runs as a non-root user and starts the `ragregress` CLI
  - dependency layers come before code layers (build cache) and torch is the CPU build
  - Dockerfile, compose file and code agree on paths (/work, artifacts/chroma, /hf-cache)
  - the API is published on 127.0.0.1 only, starts with `docker compose up`, runs the installed package and
    has a healthcheck that does not depend on Ollama

The real check is `docker compose build` plus the smoke commands in docker-compose.yml.

Run (repo root):  python -m pytest tests/unit/test_docker_files.py -v
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"
BUILTIN_SYSTEM = ROOT / "ragregress" / "rag" / "builtin_system.py"

SECRET_WORDS = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
SECRET_VALUES = re.compile(r"sk-[A-Za-z0-9]{8,}|[A-Za-z0-9_\-]{32,}")
ALLOWED_CONTEXT = {"requirements.txt", "pyproject.toml", "ragregress/", "ragregress/**"}


# ---------------------------------------------------------------- helpers

def dockerfile_instructions() -> List[Tuple[str, str]]:
    """(INSTRUCTION, arguments) in file order: comments dropped, `\\` continuations joined."""
    lines, joined = [], ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not joined and (not stripped or stripped.startswith("#")):
            continue
        if stripped.startswith("#"):                 # a comment line inside a continued instruction
            continue
        if stripped.endswith("\\"):
            joined += stripped[:-1].rstrip() + " "
            continue
        lines.append((joined + stripped).strip())
        joined = ""
    out = []
    for line in lines:
        name, _, args = line.partition(" ")
        out.append((name.upper(), args.strip()))
    return out


def instructions(name: str) -> List[str]:
    return [args for n, args in dockerfile_instructions() if n == name]


def index_of(predicate) -> int:
    for i, (name, args) in enumerate(dockerfile_instructions()):
        if predicate(name, args):
            return i
    raise AssertionError("no matching Dockerfile instruction")


def ignore_rules() -> List[str]:
    return [line.strip() for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def service(compose) -> dict:
    return compose["services"]["ragregress"]          # the CLI


@pytest.fixture(scope="module")
def api(compose) -> dict:
    return compose["services"]["api"]


def env_of_dockerfile() -> dict:
    values = {}
    for args in instructions("ENV"):
        for pair in re.findall(r"(\w+)=(\S+)", args):
            values[pair[0]] = pair[1]
    return values


# ---------------------------------------------------------------- Dockerfile

def test_base_image_matches_the_supported_python():
    (base,) = instructions("FROM")
    assert base.startswith("python:3.11"), base
    assert "requires-python = \">=3.11\"" in PYPROJECT.read_text(encoding="utf-8")


def test_the_image_runs_as_a_non_root_user():
    users = instructions("USER")
    assert users and users[-1] not in {"root", "0", "0:0", "root:root"}
    assert any("useradd" in args for args in instructions("RUN"))


def test_the_entrypoint_is_the_ragregress_cli_in_exec_form():
    assert instructions("ENTRYPOINT") == ['["ragregress"]']
    assert instructions("CMD") == ['["--help"]']


def test_the_final_working_directory_is_work():
    assert instructions("WORKDIR")[-1] == "/work"


def test_dependencies_are_installed_before_the_code_is_copied():
    copy_requirements = index_of(lambda n, a: n == "COPY" and a.startswith("requirements.txt"))
    install_requirements = index_of(lambda n, a: n == "RUN" and "pip install -r requirements.txt" in a)
    copy_code = index_of(lambda n, a: n == "COPY" and a.startswith("ragregress"))
    assert copy_requirements < install_requirements < copy_code


def test_torch_is_the_cpu_build_taken_from_the_requirements_pin_and_installed_first():
    torch_step = index_of(lambda n, a: n == "RUN" and "download.pytorch.org/whl/cpu" in a)
    requirements_step = index_of(lambda n, a: n == "RUN" and "pip install -r requirements.txt" in a)
    run = dockerfile_instructions()[torch_step][1]
    assert "--index-url https://download.pytorch.org/whl/cpu" in run
    assert "grep -E '^torch==' requirements.txt" in run            # one pin only: no version typed here
    assert torch_step < requirements_step                          # own RUN: a failed 2nd step keeps the cached torch layer
    assert "--extra-index-url" not in DOCKERFILE.read_text(encoding="utf-8")   # the CPU index must not mix with PyPI
    assert re.search(r"^torch==\S+$", REQUIREMENTS.read_text(encoding="utf-8"), re.MULTILINE)


def test_the_requirements_install_survives_a_flaky_network():
    run = next(a for a in instructions("RUN") if "pip install -r requirements.txt" in a)
    assert "--mount=type=cache,target=/root/.cache/pip" in run      # downloads kept between attempts and builds
    assert "unset PIP_NO_CACHE_DIR" in run                          # else the ENV above would switch the cache off
    assert re.search(r"for \w+ in (\d+ )+\d+; do", run) and "exit 1" in run   # retried, and still fails if all attempts fail
    assert "exit 0" in run
    assert not re.search(r"\|\|\s*true", run)                      # a failure must not be swallowed


def test_pip_is_patient_on_slow_networks():
    env = env_of_dockerfile()
    assert int(env["PIP_DEFAULT_TIMEOUT"]) >= 60
    assert int(env["PIP_RETRIES"]) >= 5


def test_the_package_is_installed_without_dependencies_after_its_files_are_copied():
    copy_code = index_of(lambda n, a: n == "COPY" and a.startswith("ragregress"))
    copy_pyproject = index_of(lambda n, a: n == "COPY" and a.startswith("pyproject.toml"))
    install = index_of(lambda n, a: n == "RUN" and "pip install --no-deps ." in a)
    assert max(copy_code, copy_pyproject) < install
    editable = [a for a in instructions("RUN") if re.search(r"pip install\s+(-\S+\s+)*-e\b", a)]
    assert not editable                                              # a real install: packaging bugs must show up here


def test_only_files_the_context_allows_are_copied():
    sources = []
    for args in instructions("COPY"):
        parts = args.split()
        assert not [p for p in parts if p.startswith("--from")], "multi-stage copies are not expected here"
        sources += parts[:-1]
    assert sources, "no COPY found"
    allowed = {s.rstrip("/") for s in ALLOWED_CONTEXT if not s.endswith("**")}
    assert {s.rstrip("/") for s in sources} <= allowed, sources


def test_no_add_no_copy_of_the_whole_context_no_secret_files():
    assert not instructions("ADD")
    for args in instructions("COPY"):
        assert ".env" not in args
        assert not re.match(r"^\.(/)?\s", args), "COPY . would send everything the context allows"


def test_no_secret_in_env_or_arg():
    for name in ("ENV", "ARG"):
        for args in instructions(name):
            for key, value in re.findall(r"(\w+)=?(\S*)", args):
                assert not SECRET_WORDS.search(key), f"{name} {key}: secrets never go into an image"
                assert not SECRET_VALUES.search(value), f"{name} {key}: looks like a credential"


def test_hf_home_and_the_chroma_mount_point_exist_and_belong_to_the_app_user():
    assert env_of_dockerfile()["HF_HOME"] == "/hf-cache"
    run = " ".join(instructions("RUN"))
    assert "mkdir -p /work/artifacts/chroma /hf-cache" in run
    assert re.search(r"chown -R app:app [^&]*/work[^&]*/hf-cache|chown -R app:app [^&]*/hf-cache[^&]*/work", run)


def test_python_is_unbuffered_and_writes_no_bytecode():
    env = env_of_dockerfile()
    assert env["PYTHONUNBUFFERED"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"


# ---------------------------------------------------------------- .dockerignore

def test_the_build_context_is_an_allow_list():
    rules = ignore_rules()
    assert rules[0] == "*", "the first rule must exclude everything"
    included = {r[1:] for r in rules if r.startswith("!")}
    assert included == ALLOWED_CONTEXT, included


def test_secrets_and_state_can_never_be_re_included():
    included = " ".join(r for r in ignore_rules() if r.startswith("!"))
    for forbidden in (".env", ".db", "artifacts", "reports", "data", ".git", ".rag_lab_env"):
        assert forbidden not in included, forbidden


def test_build_debris_is_excluded_after_the_allow_rules():
    rules = ignore_rules()
    last_allow = max(i for i, r in enumerate(rules) if r.startswith("!"))
    for pattern in ("**/__pycache__", "**/*.pyc"):
        assert pattern in rules and rules.index(pattern) > last_allow, pattern


def test_every_copy_source_is_allowed_by_the_ignore_file():
    included = {r[1:].rstrip("/") for r in ignore_rules() if r.startswith("!")}
    for args in instructions("COPY"):
        for source in args.split()[:-1]:
            assert source.rstrip("/") in included, f"COPY {source} would find nothing in the build context"


# ---------------------------------------------------------------- docker-compose.yml

def test_compose_builds_this_directory_and_names_the_image(service):
    assert service["build"]["context"] == "."
    assert service["image"] == "ragregress:local"


def test_compose_mounts_the_repo_root_and_the_two_named_volumes(service, compose):
    assert service["working_dir"] == "/work"
    assert service["volumes"] == [".:/work", "chroma_index:/work/artifacts/chroma", "hf_cache:/hf-cache"]
    assert set(compose["volumes"]) == {"chroma_index", "hf_cache"}


def test_compose_reaches_ollama_on_the_host(service):
    assert service["environment"]["OLLAMA_HOST"] == "http://host.docker.internal:11434"
    assert service["extra_hosts"] == ["host.docker.internal:host-gateway"]


def test_compose_has_no_secrets_no_env_file_no_ports(service):
    assert "env_file" not in service and "ports" not in service
    for key, value in service["environment"].items():
        assert not SECRET_WORDS.search(key), key
        assert not SECRET_VALUES.search(str(value)), key
    assert "sk-" not in COMPOSE.read_text(encoding="utf-8")


def test_compose_signals_reach_the_cli(service):
    assert service["init"] is True                                    # Ctrl+C stops a run


def test_compose_and_dockerfile_agree_on_the_hf_cache_and_the_work_directory(service):
    targets = {v.split(":")[1] for v in service["volumes"]}
    assert env_of_dockerfile()["HF_HOME"] in targets
    assert "/work" in targets and instructions("WORKDIR")[-1] == service["working_dir"]


def test_the_chroma_volume_matches_the_index_location_in_the_code():
    if not BUILTIN_SYSTEM.is_file():
        pytest.skip("ragregress/rag/builtin_system.py not in this tree")
    match = re.search(r'persist_directory:\s*str\s*=\s*"([^"]+)"', BUILTIN_SYSTEM.read_text(encoding="utf-8"))
    assert match, "BuiltinRAGSystem no longer has a persist_directory default: update docker-compose.yml"
    assert f"/work/{match.group(1)}" == "/work/artifacts/chroma"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_docker_compose_config_accepts_the_file():
    result = subprocess.run(["docker", "compose", "-f", str(COMPOSE), "config", "--quiet"],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------- docker-compose.yml: the API service (B4)

def test_compose_has_exactly_the_api_and_the_cli_services(compose):
    assert set(compose["services"]) == {"api", "ragregress"}


def test_the_api_starts_with_docker_compose_up_and_the_cli_does_not(api, service):
    assert "profiles" not in api                       # no profile: `up` starts it
    assert service["profiles"] == ["cli"]              # `run` still works for a service in a profile


def test_the_api_port_is_published_on_loopback_only(api, compose):
    assert api["ports"] == ["127.0.0.1:8000:8000"]
    published = [p for svc in compose["services"].values() for p in svc.get("ports", [])]
    assert published == ["127.0.0.1:8000:8000"]        # nothing else is published, by any service
    for port in published:
        assert port.startswith("127.0.0.1:"), "a port without the host address is open to the whole network"


def test_the_api_runs_uvicorn_on_the_app_in_the_installed_package(api):
    assert api["entrypoint"] == ["uvicorn"]
    command = api["command"]
    assert command[0] == "ragregress.main:app"
    assert command[command.index("--host") + 1] == "0.0.0.0"           # reachable through the published port
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--app-dir") + 1] == "/app"           # not /work: the mounted source must not shadow the install
    assert "--reload" not in command


def test_the_container_port_matches_the_command_and_the_healthcheck(api):
    command = api["command"]
    container_port = api["ports"][0].rsplit(":", 1)[1]
    assert command[command.index("--port") + 1] == container_port
    assert f"127.0.0.1:{container_port}/health" in " ".join(api["healthcheck"]["test"])


def test_both_services_share_the_image_the_build_and_the_mounts(api, service):
    for key in ("build", "image", "working_dir", "init", "environment", "extra_hosts", "volumes"):
        assert api[key] == service[key], key


def _seconds(value: str) -> float:
    assert value.endswith("s"), value
    return float(value[:-1])


def test_the_healthcheck_asks_the_api_not_ollama_and_waits_long_enough(api):
    health = api["healthcheck"]
    test = health["test"]
    assert test[0] == "CMD" and test[1] == "python"                    # the slim image has no curl
    assert "/health" in " ".join(test)
    timeout_in_command = float(re.search(r"timeout=(\d+(?:\.\d+)?)", " ".join(test)).group(1))
    assert timeout_in_command >= 6                                     # /health can take ~4 s when Ollama is down
    assert _seconds(health["timeout"]) > timeout_in_command
    assert health["retries"] >= 2 and _seconds(health["start_period"]) >= 10


def test_the_api_is_not_restarted_behind_your_back(api):
    assert api.get("restart", "no") == "no"            # a restart would silently kill a run in progress


def test_the_api_service_has_no_secrets_and_no_env_file(api):
    assert "env_file" not in api
    for key, value in api["environment"].items():
        assert not SECRET_WORDS.search(key), key
        assert not SECRET_VALUES.search(str(value)), key


def test_the_api_dependencies_are_pinned_in_the_requirements():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    for package in ("fastapi", "uvicorn"):
        assert re.search(rf"^{package}==\S+", text, re.MULTILINE), package


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_docker_compose_up_would_start_only_the_api():
    def services(*extra):
        result = subprocess.run(["docker", "compose", "-f", str(COMPOSE), *extra, "config", "--services"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return set(result.stdout.split())
    assert services() == {"api"}
    assert services("--profile", "cli") == {"api", "ragregress"}