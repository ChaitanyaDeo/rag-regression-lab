# Security and robustness

What RAG Regression Lab protects against, how, which tests prove it, and which risks are known and accepted. Architecture: [architecture.md](architecture.md). Usage: [README](../README.md).

## Contents

1. [Threat model](#1-threat-model)
2. [What is tested](#2-what-is-tested)
3. [Mitigations in place](#3-mitigations-in-place)
4. [Secrets and API keys](#4-secrets-and-api-keys)
5. [Logs and redaction](#5-logs-and-redaction)
6. [The API and Docker](#6-the-api-and-docker)
7. [Known risks and accepted behaviour](#7-known-risks-and-accepted-behaviour)
8. [Secrets check script](#8-secrets-check-script)
9. [Follow-ups](#9-follow-ups)

## 1. Threat model

The project is local-first: Ollama models on the user's machine, no third-party service and no API key by default, an API bound to localhost. The realistic risks are therefore:

| # | Threat | Example |
|---|---|---|
| T1 | Bad or hostile configuration | a typo that silently changes an experiment; a YAML tag that tries to run code; absurd values |
| T2 | Bad judge output | malformed JSON, missing fields, out-of-range verdicts that could inflate or hide a score |
| T3 | Untrusted text in passages | prompt injection through retrieved context |
| T4 | Secrets or personal data leaking | an API key committed to git, written to a log, stored in the database or returned by the API |
| T5 | Unintended network exposure | the API reachable from the local network |

## 2. What is tested

Counts from the full suite (871 passed, 3 skipped, 5 xfailed; Nov 2, 2026).

| Area | File | Tests | What it proves |
|---|---|---|---|
| Config validation | `tests/security/test_config_validation.py` | 34 | missing, oversized, negative values, unknown keys, unknown pipeline id and an `is_dev` / split mismatch are rejected at load (exit 3); a YAML `!!python` tag is not executed (`safe_load`) |
| Malformed judge, client | `tests/security/test_malformed_judge.py` | 21 (16 pass, 5 xfail) | real `instructor` against a fake Ollama server: garbage makes exactly `max_retries + 1` requests then raises; later recovery works; schema-invalid replies raise; seed and temperature reach the wire. The 5 xfails pin S2 below. |
| Malformed judge, pipeline | `tests/security/test_malformed_judge_pipeline.py` | 11 | scripted judge failures give `judge_failed` rows; NaN rows stored with a reason; aggregates ignore NaN; validity below 80 % makes the comparison INCONCLUSIVE |
| Log redaction | `tests/security/test_log_redaction.py` | 34 | `redact()` and `RedactingFilter` remove e-mails, `sk-` / `ghp_` / `hf_` keys, Bearer tokens and `key=value` secrets; run ids, hashes and passage ids stay readable |
| Logging through real handlers | `tests/unit/test_logging.py`, `tests/integration/test_pipeline_logging.py` | 24, 21 | the console handler and the per-run log file emit redacted JSON with the schema fields |
| Judge configuration | `tests/unit/test_judge.py` | 125 | provider, endpoint and key resolution from the environment and `.env` |
| Secrets exposure, logic | `tests/security/test_secrets_check.py` | 43 | `scripts/check_secrets.py` on throw-away git repos: `.gitignore` gaps, bad `.env.example`, secret shapes in tracked files and in deleted-but-committed history; masked output |
| Secrets exposure, guard | `tests/security/test_secrets_exposure.py` | 3 | on the real repo: `.env*` ignored, `.env.example` placeholders only, no tracked secret shape |
| API | `tests/integration/test_api.py` | 77 | every endpoint's status codes and input validation (run ids, config names, candidate lists) |
| Docker files | `tests/unit/test_docker_files.py` | 37 | Dockerfile and Compose structure |

Fake secrets in tests are assembled at runtime (`"sk" + "-..."`), so the test files themselves contain no secret shapes. Tests never write to the real `ragregress.db` or `artifacts/chroma`.

## 3. Mitigations in place

| Threat | Mitigation | Evidence |
|---|---|---|
| T1 | Pydantic v2 models, `extra="forbid"` and frozen at every level; caps (`top_k` 1–50, `chunk_size` 1–4000, `overlap` < `chunk_size`, `max_workers` 1–8, `reranker.candidates` ≤ 50 and ≥ `top_k`, seeds ≥ 0); YAML `safe_load` | `test_config_validation.py` |
| T1 | `POST /runs` takes a config **name** matched against `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$` and resolved inside `configs/`, never a path; the config's `db_path` must equal the API's database | `api/schemas.py`, `api/experiments.py`, `test_api.py` |
| T2 | bounded retries, then the metric is recorded as failed for that question (`judge_failed`) | `test_malformed_judge*.py` |
| T2 | unusable judge output gives NaN or an absent metric; both count as unmeasured; below 80 % validity the verdict is INCONCLUSIVE, so a broken judge cannot pass a change | `test_malformed_judge_pipeline.py` |
| T4 | judge key from the environment only; never hashed, stored or logged; excluded from `repr` | `evaluation/judge.py` (`ResolvedJudge.api_key` has `repr=False`) |
| T4 | `.gitignore` covers `.env`, `.env.*` (except `.env.example`), the database and `artifacts/`; `check_secrets.py` plus guard tests | `test_secrets_*.py` |
| T4 | every log record passes through redaction; API error text is redacted before it is returned | `core/logging.py`, `test_log_redaction.py`; `api/experiments.py`, `api/jobs.py` |
| T5 | API port published on `127.0.0.1` only | `docker-compose.yml` (`127.0.0.1:8000:8000`) |
| (report) | Jinja2 autoescape: answers and passages are untrusted text and cannot inject HTML or script; the report loads no external assets | `report.py`, `test_report.py` |

## 4. Secrets and API keys

| Item | Rule |
|---|---|
| Default setup | no key at all (local Ollama). The OpenAI-compatible client gets a placeholder value where a key is required. |
| Hosted judge | `judge_provider: openai` reads `JUDGE_API_KEY` (or `OPENAI_API_KEY`) from the environment or `.env` |
| Where a key can appear | only in the process environment and the HTTP request to the judge |
| Where it never appears | config files, `config_hash`, the database, logs, API responses, `--dry-run` output (which prints where the key came from, not its value) |
| A leaked key | rotate it; removing it from git history is not enough |

## 5. Logs and redaction

`ragregress/core/logging.py` is wired into every entry point (CLI, API, per-run log file).

| Property | Detail |
|---|---|
| Redacted | e-mail addresses; `sk-…`, `ghp_` / `gho_` / `ghu_` / `ghs_` / `ghr_…`, `hf_…` key shapes; `Bearer <token>`; values of `api_key`, `token`, `password`, `passwd`, `secret` in `key=value`, `key: value` and JSON form |
| How | `RedactingFilter` on each handler redacts the message and exception text; the JSON formatter redacts every string value before serialising (so the JSON stays valid) |
| Not redacted, on purpose | hex strings in general (config hashes, run ids, passage ids must stay readable) |
| Not redacted | phone numbers, IP addresses, user paths; `stack_info` text; output that bypasses `logging` |
| Over-redaction | possible by design (`token=abc` is redacted) |

What the logs **do** contain:
- **Corpus and question text.** Run logs can include question text (e.g. the generator's "no citations" warning) and, in exception tracebacks, the judge's full replies. Harmless for SQuAD; relevant if the harness is pointed at a private corpus.
- **Third-party records.** The per-run log file attaches to the root logger, so records from libraries (e.g. `instructor` retry errors) are written too, redacted but without the pipeline fields.

## 6. The API and Docker

| Item | Behaviour |
|---|---|
| Authentication | **none**. Acceptable only because the port is bound to `127.0.0.1`; do not publish it on other interfaces. |
| Inside the container | uvicorn listens on `0.0.0.0` (required to be reachable through the published port) |
| Input validation | run ids `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`, config names as above, 1–10 unique compare candidates, unknown JSON keys rejected (422) |
| Error responses | passed through `redact()`; `/runs` lists no error text |
| Background runs | one at a time per process; job errors redacted before storage |
| Image | `python:3.11-slim`, non-root user (uid 1000), `.dockerignore` allow-list (no `.env`, database, index or data in the image), pinned requirements |
| Secrets in containers | `.env` is read from the bind-mounted repo at runtime, never baked into the image |

## 7. Known risks and accepted behaviour

| ID | Risk or behaviour | Status | Plan |
|---|---|---|---|
| S1 | **Prompt injection.** Retrieved chunks are pasted into the generator prompt; a passage can try to steer the answer. | **Mitigated, not solved.** Prompt `v2` (opt-in via `pipeline.params.prompt`; `v1` stays the default so stored runs and hashes are unchanged) fences passages in tags, states that they are data, and neutralises fence tags inside passage text. Tests: `tests/unit/test_prompts.py` (fencing, escaping, frozen prompt texts) and `tests/security/test_prompt_injection.py` (runner with a fake model; opt-in real-model test, `RAGREGRESS_SLOW=1`). `scripts/injection_check.py` prints the hijack rate per prompt on six cases. Measured (qwen2.5:3b-instruct, 6 cases x 6 seeds): v1 hijacked 7/36, v2 6/36; v2 fixes `ignore_previous` and `fake_system_message` but is hijacked by `delimiter_breakout` on every seed (v1 is not). Kept as a documented finding, not fixed. The check reuses the prompt's own example fact, so it is a rough guide. Residual risk: a small model can still obey an injection; look-alike Unicode brackets are not neutralised; a custom prompt that uses `{context}` gets no fencing. | Judge prompts: see S12. |
| S2 | **Judge verdict range.** Fixed Nov 3: `verdict` accepts only 0 or 1 (`int` with `ge=0, le=1`). A value like 2 or −1 is rejected by the schema, retried, and ends as `judge_failed`, so it can no longer inflate faithfulness. | Resolved; covered by the former xfail tests and `test_verdict_schema.py`. | None. |
| S3 | **NaN faithfulness row tagged `pass`.** No claims, or a verdict-count mismatch, gives NaN without an exception. | Accepted: NaN is unmeasured in comparisons and validity protects the verdict. | Optional: a distinct tag. |
| S4 | **Same-count mismatched verdicts.** A verdict list of the right length but for different statements is accepted. | Untested. | Match verdicts to statements, or document as a judge-quality limit. |
| S5 | **Lax integer coercion** in configs (`true`, `"5"`, `5.0` accepted as ints; `5.5`, `"1e3"` rejected). | Accepted and pinned. | None. |
| S6 | **YAML alias bomb** (nested anchors that expand exponentially). `safe_load` blocks code execution, not expansion. | Untested; configs are written by the user. | Size cap on config files if configs ever come from untrusted sources. |
| S7 | **NaN reasons are persisted** in `metric_results_json`. | Intended (debuggability). | None. |
| S8 | **Raw error text in SQLite.** `last_error` and `run_status_detail` are stored unredacted (exception text from libraries). | Accepted: the database is local and gitignored; the API redacts both on output. | Redact on write if the database is ever shared. |
| S9 | **No API authentication.** | Accepted for a localhost-only tool (T5 mitigation). | Add authentication before any non-local deployment. |
| S10 | **Logs contain corpus text and judge output** ([§5](#5-logs-and-redaction)). | Accepted for a public corpus. | Treat logs as sensitive when using private data. |
| S11 | **Run guard is per process.** A CLI run and an API run can write to the same database at once. | Accepted: SQLite serialises writes and runs have distinct ids; not stress-tested. | A file lock if needed. |
| S12 | **Prompt injection against the judge.** The answer and passages are pasted into the judge prompts as plain text, so a hijacked answer can try to steer its own faithfulness verdict. | Untested; the generator prompt can be hardened (S1), the judge prompts are not. | Move the judge prompts to versioned files like the generator's; add judge-side cases to the injection check. |

## 8. Secrets check script

`python scripts/check_secrets.py` (standard library plus `git`; exit 0 clean, 1 findings, 2 cannot run):

| Check | What it does |
|---|---|
| gitignore | `.env`, `.env.local`, `.env.production`, `.env.development` ignored; `.env.example` not ignored |
| env-example | placeholders only (no secret shapes, no real-looking values) |
| tracked | no tracked file has a secret shape; no tracked `.env*` except the example |
| env-history | no `.env*` (except the example) ever committed |
| history | no commit on any branch ever added a secret-shaped line |

Limits:
- Only tracked files and committed history are scanned; new, uncommitted files are not.
- Shapes only (`sk-`, `ghp_` family, `hf_`, `AKIA`, Slack `xox`, private-key headers). A secret without a recognisable shape outside a `.env` file is not found.
- Findings are masked (first 4 characters and length), so the report is safe to paste.
- Run it before publishing the repository; it grows with the history, so it is a manual check, not part of the suite.

## 9. Follow-ups

| Item | Risk |
|---|---|
| `prompt`, prompt hardening, prompt-injection test | S1 (done; judge side: S12) |
| `verdict: Literal[0, 1]`, remove the 5 xfail markers | S2 |
| Config file size cap | S6 |
| Verdict-to-statement matching; distinct tag for NaN rows | S4, S3 |
| Redact `last_error` / `run_status_detail` on write | S8 |
| Authentication, if the API is ever exposed beyond localhost | S9 |