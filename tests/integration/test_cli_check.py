"""
tests/integration/test_cli_check.py


"""

from __future__ import annotations
 
from typing import Any, Dict, Iterable, List, Optional
 
import pytest
from typer.testing import CliRunner

from pathlib import Path
from ragregress import cli
from ragregress.cli import app
from ragregress.storage.database import init_db
from ragregress.storage.models import QueryResult, Run
from ragregress.storage.repositories import Repository

from ragregress.core.config import load_run_config
from ragregress.core.runner import build_rag_system
 
runner = CliRunner()

ROOT = Path(__file__).resolve().parents[2]


def test_keyword_overlap_check():

    config = ROOT / "configs" / "keyword_overlap.yaml"

    res = runner.invoke(app, ["check", "--config", str(config), "--debug"],)

    assert res.exit_code==0


"""TODO:
Case	                                                Then
configs/keyword_overlap.yaml, run from the repo root	exit 0
temp-module pipeline returning bad ids	                exit 4; the id is in the output
temp-module pipeline whose generate always raises	    exit 5
openai judge provider configured with no key	        still exit 0 (the judge is never touched); compare test_cli_metrics.py line 51 for the same idea [V]
"""
