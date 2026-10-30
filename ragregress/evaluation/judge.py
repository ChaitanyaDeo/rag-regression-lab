"""
ragregress/evaluation/judge.py
 
Which model judges faithfulness, on which service, and how to reach it.

The run config says WHAT (the model); the environment says WHERE (endpoint and key):

    judge_model: qwen2.5:7b-instruct          bare name = ollama   (every config written so far)
    judge_model: ollama/qwen2.5:7b-instruct   same thing, explicit
    judge_model: openai/gpt-4o-mini           OpenAI, or any OpenAI-compatible service

    provider   endpoint                                           API key
    ollama     JUDGE_BASE_URL, else OLLAMA_HOST + /v1,            not used (Ollama needs none)
               else http://localhost:11434/v1
    openai     JUDGE_BASE_URL, else OPENAI_BASE_URL,              JUDGE_API_KEY, else OPENAI_API_KEY
               else https://api.openai.com/v1                     (none needed if a base URL is set)

Any OpenAI-compatible service (vLLM, LM Studio, OpenRouter, a protected Ollama, ...) is
`openai/<model>` plus JUDGE_BASE_URL (and JUDGE_API_KEY if it wants one).

Rules
  - Only the model string is part of the run config, so only it is hashed and stored.
    The endpoint and the key come from the environment and are never hashed, stored or logged.
  - ResolvedJudge.api_key is excluded from repr(), so it cannot reach a log line by accident.
  - Providers are ours to implement. This module is not a plug-in point: to judge with a
    different service, add it here (D13 J5).
  - Nothing here touches the network; reachability is the API health check's job.

Errors: a bad `judge_model` string raises ValueError from parse_judge_model (RunConfig turns it
into a config error at load time, exit 3). A missing key or an unusable endpoint raises
ConfigValidationError from resolve_judge (exit 3, before any model is loaded).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from typing import Any, Literal, Mapping, Optional, get_args
from urllib.parse import urlsplit, urlunsplit

from ragregress.core.errors import ConfigValidationError

JudgeProvider = Literal["ollama", "openai"]
PROVIDERS = get_args(JudgeProvider)
DEFAULT_PROVIDER = "ollama"
 
BASE_URL_ENV = "JUDGE_BASE_URL"
API_KEY_ENV = "JUDGE_API_KEY"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
 
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_PORT = 11434
OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
 
# The OpenAI client refuses an empty key; servers behind a bare base URL (vLLM, LM Studio) ignore it.
PLACEHOLDER_KEY = "unused"

UNSUPPORTED_PROVIDERS = frozenset({
    "anthropic", "google", "gemini", "genai", "vertexai", "bedrock", "azure", "azure_openai",
    "groq", "mistral", "cohere", "xai", "deepseek", "together", "fireworks", "perplexity",
    "openrouter", "litellm", "cerebras", "writer", "databricks", "anyscale",
})

@dataclass(frozen=True)
class ResolvedJudge:
    """A provider and model plus the connection settings the environment gave them."""
 
    provider: str
    model: str
    base_url: Optional[str]                      # handed to the client; None = the SDK's own default
    api_key: Optional[str] = field(repr=False)   # never printed, never logged
    endpoint: str = ""                           # for display: no credentials, no query string
    key_note: str = ""                           # for display: where the key came from, never its value

# --------------------------------------------------- model string

def check_judge_provider(value: str) -> str:
    """Return the provider in canonical (lower-case) form. Raises ValueError for anything unsupported."""
    provider = value.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"unsupported judge_provider {value!r} (supported: {', '.join(PROVIDERS)}). "
            f"For another OpenAI-compatible service use 'openai' and set {BASE_URL_ENV}"
        )
    return provider
 
 
def check_judge_model(value: str) -> None:
    """Raises ValueError if `value` is empty.
 
    The provider-prefix rule ('openai/gpt-4o-mini') depends on WHERE the judge runs, so it lives in
    
    """
    if not value.strip():
        raise ValueError("judge_model is empty")

# ------------------------------------------------------------------

def _public(url: str):
    """The URL without user:password@ and without a query string, safe to print."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))

def _ollama_v1(host: Optional[str]) -> str:
    raw = (host or "").strip() or DEFAULT_OLLAMA_HOST
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if not parts.hostname or port == -1:
        raise ConfigValidationError(f"{OLLAMA_HOST_ENV}={host!r} is not a valid host or URL", source="environment")
    netloc = parts.netloc if port is not None or parts.scheme != "http" else f"{parts.netloc}:{OLLAMA_DEFAULT_PORT}"
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}/v1"

def _clean(env: Mapping[str, str], name: str) -> Optional[str]:
    return (env.get(name) or "").strip() or None

OPENAI_HOST = "api.openai.com"

def _provider_prefix(model: str) -> Optional[str]:
    """'openai' for 'openai/gpt-4o' (any case, spaces ignored); None for any other model name."""
    head, slash, _ = model.strip().partition("/")
    prefix = head.strip().lower()
    return prefix if slash and prefix in PROVIDERS else None


def _check_prefixed_model(model: str, provider: str, openai_url: Optional[str]) -> None:
    """Provider and model are separate fields: 'openai/gpt-4o-mini' is a mistake ... except through an
    aggregator. 
    
    With judge_provider: openai and a base URL that is not api.openai.com (OpenRouter, ...),
    the service's own ids may start with a provider name ('openai/gpt-4o'), so they pass untouched.
    
    """
    prefix = _provider_prefix(model)
    if prefix is None:
        return
    if provider == "openai" and openai_url and urlsplit(openai_url).hostname != OPENAI_HOST:
        return
    if prefix != provider:
        fix = f"set judge_provider: {prefix} and keep only the model name in judge_model"
    else:
        fix = "keep only the model name in judge_model"

    raise ConfigValidationError(
        f"judge_model {model!r} starts with the provider name {prefix!r}: provider and model are separate fields; "
        f"{fix}. (A prefixed id is accepted only with judge_provider: openai and a non-OpenAI endpoint in "
        f"{BASE_URL_ENV}, e.g. OpenRouter.)",
        source="judge_model",
    )


def resolve_judge(provider: str, model: str, env: Optional[Mapping[str, str]] = None) -> ResolvedJudge:
    """
        Combine the config's provider and model with the environment. 
        Raises ConfigValidationError.
    """
    env = os.environ if env is None else env
    try:
        provider = check_judge_provider(provider)
    except ValueError as exc:
        raise ConfigValidationError(str(exc), source="judge_provider") from None
    
    try:
        check_judge_model(model)
    except ValueError as exc:
        raise ConfigValidationError(str(exc), source="judge_model") from None
    
    model = model.strip()
 
    explicit_url = _clean(env, BASE_URL_ENV)
    explicit_key = _clean(env, API_KEY_ENV)
    _check_prefixed_model(model, provider, explicit_url or _clean(env, OPENAI_BASE_URL_ENV))
 
    if provider == "ollama":
        base_url = explicit_url or _ollama_v1(env.get(OLLAMA_HOST_ENV))
        note = "not used (ollama needs none)"
        if explicit_key:
            note = (f"{API_KEY_ENV} ignored: the ollama provider sends no key "
                    f"(use judge_provider: openai with {BASE_URL_ENV} for a protected server)")
        return ResolvedJudge("ollama", model, base_url, None, _public(base_url), note)
 
    # openai
    env_url = _clean(env, OPENAI_BASE_URL_ENV)
    key, key_name = (explicit_key, API_KEY_ENV) if explicit_key else (_clean(env, OPENAI_API_KEY_ENV), OPENAI_API_KEY_ENV)
    effective_url = explicit_url or env_url

    if key:
        note = f"set (from {key_name})"
    elif effective_url:
        key, note = PLACEHOLDER_KEY, "none set; a placeholder is sent (fine for a local server)"
    else:
        raise ConfigValidationError(
            f"judge provider 'openai' needs an API key: set {API_KEY_ENV} (or {OPENAI_API_KEY_ENV}) "
            f"in the environment or .env, or set {BASE_URL_ENV} for a server that needs none",
            source="environment",
        )
    
    return ResolvedJudge("openai", model, explicit_url, key, _public(effective_url or OPENAI_DEFAULT_URL), note)

# ---------------------------------------------------------------- building the judge
 
def build_judge(provider: str, model: str, seed: Optional[int], max_retries: int = 3,
                env: Optional[Mapping[str, str]] = None) -> Any:
    """The StructuredLLM the faithfulness metric uses: our instructor-based judge, pointed where the env says."""
    resolved = resolve_judge(provider, model, env)
    from ragregress.evaluation import structured_llm   # looked up at call time
 
    return structured_llm.InstructorStructuredLLM(
        resolved.model,
        provider=resolved.provider,
        max_retries=max_retries,
        seed=seed,
        base_url=resolved.base_url,
        api_key=resolved.api_key,
    )