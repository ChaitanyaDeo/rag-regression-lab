"""
ragregress/api/health.py

GET /health -> {"status": "ok", "ollama_reachable": bool, "judge_reachable": bool}
Always 200 while the API is up; the two flags only report what the API can reach.

ollama_base(env)     OLLAMA_HOST -> "http://host:port" (default http://localhost:11434, no trailing "/").
probe(url, timeout)  True if the server answers an HTTP GET (any status). False on refusal, timeout,
                     bad URL, or a scheme other than http/https.
check_health(env)    Probes Ollama at /api/tags. Probes JUDGE_BASE_URL + /models if set,
                     otherwise judge_reachable copies ollama_reachable. env=None means os.environ.
                     No credentials are sent and nothing from the environment is returned.

The endpoint is a plain `def` because urllib blocks; FastAPI runs it in a thread pool.
"""

from __future__ import annotations

import os

from typing import Mapping, Optional
import urllib.error
from urllib import parse, request

from fastapi import APIRouter


from ragregress.api.schemas import Health


DEFAULT_LOCAL_HOST = "http://localhost:11434"
DEFAULT_URL_SCHEMES = ["http", "https"]

router = APIRouter()

def ollama_base(env: Mapping[str, str]) -> str:
    ollama_host = env.get("OLLAMA_HOST", "").strip()

    if not ollama_host:
        ollama_host = DEFAULT_LOCAL_HOST

    if "://" not in ollama_host:
        ollama_host = "http://" + ollama_host
    return ollama_host.rstrip('/')


def probe(url: str, timeout: float = 2.0) -> bool:
    url_split = parse.urlsplit(url)

    if url_split.scheme not in DEFAULT_URL_SCHEMES:
        return False

    try:
        with request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        # TODO: log Error
        return True
    except (OSError, ValueError):
        #TODO: Log error
        return False
    


def check_health(env: Optional[Mapping[str, str]] = None):

    if env is None:
        env = os.environ

    judge_base_url = env.get("JUDGE_BASE_URL", "").strip()

    ollama_reachable = probe(ollama_base(env) + "/api/tags")

    if judge_base_url:
        judge_reachable = probe(judge_base_url.rstrip('/') + "/models")

    else:
        judge_reachable = ollama_reachable

    return Health(status='ok', ollama_reachable=ollama_reachable, judge_reachable=judge_reachable)


@router.get("/health", response_model=Health)
def health():
    return check_health()