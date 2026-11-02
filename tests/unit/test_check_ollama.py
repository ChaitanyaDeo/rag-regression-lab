"""
tests/unit/test_check_ollama.py

Day 13 B2b: scripts/check_ollama.py (is Ollama reachable from here, e.g. from the container?).
A local fake server on 127.0.0.1 and a random port; never the real Ollama (port 11434 is asserted
not to be used). OLLAMA_HOST is set explicitly in every test that reaches the network.

Run (repo root):  python -m pytest tests/unit/test_check_ollama.py -v
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_ollama.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_ollama", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body, status = self.server.reply
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.server.paths.append(self.path)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.reply = ({"models": [{"name": "qwen2.5:3b-instruct"}, {"name": "qwen2.5:7b-instruct"}]}, 200)
    srv.paths = []
    assert srv.server_port != 11434
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("host, expected", [
    (None, "http://localhost:11434"),
    ("", "http://localhost:11434"),
    ("   ", "http://localhost:11434"),
    ("localhost:11434", "http://localhost:11434"),
    ("http://host.docker.internal:11434", "http://host.docker.internal:11434"),
    ("http://gpu-box:8080/", "http://gpu-box:8080"),
    (" https://ollama.example.org ", "https://ollama.example.org"),
])
def test_ollama_url(check, host, expected):
    assert check.ollama_url(host) == expected


def test_reachable_lists_the_models_and_exits_0(check, server, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    assert check.main() == 0
    out = capsys.readouterr().out
    assert f"ollama endpoint: http://127.0.0.1:{server.server_port}" in out
    assert "2 model(s): qwen2.5:3b-instruct, qwen2.5:7b-instruct" in out
    assert server.paths == ["/api/tags"]


def test_reachable_with_no_models_says_so(check, server, monkeypatch, capsys):
    server.reply = ({"models": []}, 200)
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{server.server_port}")      # host:port form
    assert check.main() == 0
    assert "0 model(s): (none pulled)" in capsys.readouterr().out


def test_nothing_listening_exits_1(check, monkeypatch, capsys):
    import socket
    with socket.socket() as s:                                                 # a port that is free right now
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
    assert check.main() == 1
    assert "NOT REACHABLE" in capsys.readouterr().out


@pytest.mark.parametrize("reply", [
    (b"not json", 200),
    ([1, 2, 3], 200),
    ({"models": "oops"}, 200),
    ({"error": "boom"}, 500),
])
def test_an_unusable_reply_exits_1_without_a_traceback(check, server, monkeypatch, capsys, reply):
    server.reply = reply
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    assert check.main() == 1
    assert "NOT REACHABLE" in capsys.readouterr().out


def test_entries_without_a_name_do_not_crash(check, server, monkeypatch):
    server.reply = ({"models": [{"name": "a"}, {}, "junk", {"name": 5}]}, 200)
    assert check.list_models(f"http://127.0.0.1:{server.server_port}") == ["a", "?", "5"]