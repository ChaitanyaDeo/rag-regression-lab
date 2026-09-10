# RAG Regression Lab - Day 1 repo skeleton scaffolder
# Run this from inside the empty rag-regression-lab/ folder (root of the project).
# Creates the full tree per the project plan's Section 16, as empty stub files.

$ErrorActionPreference = "Stop"

function New-Dir($path) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

function New-StubFile($path, $content = "") {
    $dir = Split-Path $path -Parent
    if ($dir) { New-Dir $dir }
    if (-not (Test-Path $path)) {
        Set-Content -Path $path -Value $content -NoNewline
    }
}

# --- ragregress/ package ---
New-StubFile "ragregress/__init__.py"
New-StubFile "ragregress/main.py" "# FastAPI app entrypoint - Day 13`n"
New-StubFile "ragregress/cli.py" "# Typer CLI: ragregress run / compare / report - Day 7`n"

New-StubFile "ragregress/api/__init__.py"
New-StubFile "ragregress/api/experiments.py" "# /runs, /compare, /reports/{id} - Day 13`n"
New-StubFile "ragregress/api/health.py" "# /health - Day 13`n"

New-StubFile "ragregress/core/__init__.py"
New-StubFile "ragregress/core/config.py" "# Pydantic RunConfig model + field constraints - Day 7 / Day 11`n"
New-StubFile "ragregress/core/logging.py" "# Structured JSON logging setup - Day 13`n"
New-StubFile "ragregress/core/errors.py" "# Shared exception types - Day 7`n"

New-StubFile "ragregress/benchmark/__init__.py"
New-StubFile "ragregress/benchmark/loader.py" "# Dataset load/cache wrapper - Day 1`n"
New-StubFile "ragregress/benchmark/schemas.py" "# Pydantic models: passage, QA pair, category - Day 1`n"

New-StubFile "ragregress/rag/__init__.py"
New-StubFile "ragregress/rag/chunker.py" "# Fixed-size chunking - Day 2`n"
New-StubFile "ragregress/rag/embedder.py" "# sentence-transformers wrapper - Day 2`n"
New-StubFile "ragregress/rag/retriever.py" "# Chroma client wrapper - Day 2`n"
New-StubFile "ragregress/rag/generator.py" "# ChatOllama generation + citations - Day 4`n"
New-StubFile "ragregress/rag/pipeline.py" "# LangGraph StateGraph - Day 6`n"
New-StubFile "ragregress/rag/interface.py" "# RAGSystem Protocol + Chunk schema - Day 2`n"
New-StubFile "ragregress/rag/builtin_system.py" "# BuiltinRAGSystem(RAGSystem) - Day 2`n"

New-StubFile "ragregress/evaluation/__init__.py"
New-StubFile "ragregress/evaluation/retrieval.py" "# Recall@K, MRR - Day 3`n"
New-StubFile "ragregress/evaluation/groundedness.py" "# Faithfulness judge + SQuAD-F1 - Day 4 / Day 5`n"
New-StubFile "ragregress/evaluation/comparator.py" "# Diff two runs - Day 8`n"
New-StubFile "ragregress/evaluation/regression.py" "# PASS/FAIL/INCONCLUSIVE thresholds - Day 8`n"

New-StubFile "ragregress/storage/__init__.py"
New-StubFile "ragregress/storage/database.py" "# SQLite schema - Day 6`n"
New-StubFile "ragregress/storage/repositories.py" "# RunRepository, MetricsRepository - Day 6`n"

New-StubFile "ragregress/report.py" "# Jinja2 HTML report renderer - Day 9`n"
New-Dir "ragregress/templates"
New-StubFile "ragregress/templates/index.html" "<!-- Regression report template - Day 9 -->`n"

# --- configs/ (empty for now) ---
New-Dir "configs"

# --- scripts/ ---
New-StubFile "scripts/download_dataset.py" "# Loads rag-mini-wikipedia, prints real column names - Day 1`n"
New-StubFile "scripts/suggest_categories.py" "# LLM-assisted CRAG category tagging - Day 1`n"
New-StubFile "scripts/run_demo.sh" "#!/usr/bin/env bash`n# Run baseline + candidate, compare, open report - Day 9`n"

# --- artifacts/reports/ ---
New-Dir "artifacts/reports"
New-StubFile "artifacts/reports/.gitkeep"

# --- data/ ---
New-StubFile "data/README.md" "# Dataset cache location`n`nDocuments where the rag-mini-wikipedia cache and data/splits/*.json land. Cache itself is gitignored.`n"

# --- tests/ ---
New-Dir "tests/unit"
New-Dir "tests/integration"
New-Dir "tests/security"

# --- docs/ ---
New-Dir "docs"

# --- top-level files ---
New-StubFile "Dockerfile" "# Day 13`n"
New-StubFile "docker-compose.yml" "# Day 13 - mounts SQLite file as a volume; Ollama runs on host, not containerized`n"
New-StubFile "pyproject.toml" "[project]`nname = `"ragregress`"`nversion = `"0.1.0`"`nrequires-python = `">=3.11`"`n`n[project.scripts]`nragregress = `"ragregress.cli:app`"`n"
New-StubFile ".env.example" "OLLAMA_HOST=http://localhost:11434`nLOG_LEVEL=INFO`n"
New-StubFile ".gitignore" ".venv/`n__pycache__/`n*.pyc`ndata/*.parquet`ndata/cache/`nartifacts/reports/*.html`nartifacts/reports/*.json`n!artifacts/reports/.gitkeep`n*.db`n.env`n"
New-StubFile "LICENSE" "# TODO: pick a license (Day 14)`n"
New-StubFile "README.md" "# RAG Regression Lab`n`n(Day 1 scaffold - content to follow per Section 22 structure)`n"

Write-Host "Repo skeleton created." -ForegroundColor Green