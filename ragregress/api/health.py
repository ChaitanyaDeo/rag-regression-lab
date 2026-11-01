"""
ragregress/api/health.py   --- YOUR FILE (B3 step 1)

GET /health  ->  {"status": "ok", "ollama_reachable": bool, "judge_reachable": bool}

Names the tests import (all three): ollama_base, probe, check_health. The endpoint calls check_health().

    ollama_base(env: Mapping[str, str]) -> str
        OLLAMA_HOST ('host:port', 'http://host:port', blank or missing) -> 'http://host:port', no trailing '/'.
        default http://localhost:11434 ; a value without '://' gets 'http://' in front ; whitespace is stripped.

    probe(url: str, timeout: float = 2.0) -> bool
        True if a server answers an HTTP GET at url. ANY HTTP answer counts (even 401 / 404).
        False for: refused connection, timeout, bad URL, and any scheme other than http / https
        (never request file:// or ftp:// : the URL comes from the environment).
        Hint: urllib.request.urlopen, urllib.error.HTTPError, OSError, ValueError, urllib.parse.urlsplit.

    check_health(env: Optional[Mapping[str, str]] = None) -> Health        # env None -> os.environ
        ollama_reachable = probe(ollama_base(env) + "/api/tags")
        judge_reachable  = probe(JUDGE_BASE_URL.rstrip("/") + "/models")  if JUDGE_BASE_URL is set (non-blank)
                           else the same value as ollama_reachable (do NOT probe twice)
        status is always "ok": the endpoint answers 200 whenever the API itself is up.
        No key is ever sent, and nothing from the environment appears in the result.

    Health: a pydantic model (put it in schemas.py): status: str, ollama_reachable: bool, judge_reachable: bool.
    The tests call check_health(...).model_dump_json() and read the three attributes.

Things to read: "Bigger Applications" (APIRouter), "Response Model". A plain `def` endpoint is right here
(urllib blocks; FastAPI runs `def` endpoints in a thread pool).
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