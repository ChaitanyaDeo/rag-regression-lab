"""
scripts/check_ollama.py

Is Ollama reachable from here? Prints the endpoint and the models it lists; exit 0 if yes, 1 if not.
Uses OLLAMA_HOST like the generator does (default http://localhost:11434). Standard library only.

    python scripts/check_ollama.py
    docker compose run --rm --entrypoint python ragregress scripts/check_ollama.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import List

DEFAULT_HOST = "http://localhost:11434"
TIMEOUT_S = 5


def ollama_url(host: str | None) -> str:
    """OLLAMA_HOST ('host:port', 'http://host:port', blank) -> 'http://host:port' without a trailing slash."""
    raw = (host or "").strip() or DEFAULT_HOST
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def list_models(base_url: str, timeout: float = TIMEOUT_S) -> List[str]:
    """Names of the models Ollama lists at /api/tags. Raises OSError / ValueError if it cannot be reached or understood."""
    with urllib.request.urlopen(base_url + "/api/tags", timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("models", []), list):
        raise ValueError("unexpected reply from /api/tags")
    return [str(m.get("name", "?")) for m in payload.get("models", []) if isinstance(m, dict)]


def main() -> int:
    base_url = ollama_url(os.environ.get("OLLAMA_HOST"))
    print(f"ollama endpoint: {base_url}")
    try:
        models = list_models(base_url)
    except (OSError, ValueError) as exc:        # URLError, timeouts, refused connections, bad JSON
        print(f"NOT REACHABLE: {exc}")
        return 1
    print(f"reachable; {len(models)} model(s): {', '.join(models) if models else '(none pulled)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())