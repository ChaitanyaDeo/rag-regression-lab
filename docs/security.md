# Security and robustness notes

Scope: what the RAG Regression Lab tests for adversarial or malformed input, which mitigations are
active, and which risks are accepted or deferred. Written on Day 11; update on Day 13 (logging
wiring) and after Day 14 (prompt hardening).

The project is local-first: Ollama models, no third-party API key, no network service. The
threat model is therefore (1) bad config, (2) bad judge output, (3) untrusted text inside
passages, (4) secrets or personal data leaking through git or logs.

## 1. What is tested

All files are in `tests/security/`. Baseline on the author's machine (Oct 26): **473 passed,
2 skipped, 5 xfailed** for `tests/unit tests/integration tests/security`.

| Area | File | Tests | What it proves |
|---|---|---|---|
| Config validation | `test_config_validation.py` | 34 | Missing/oversized/negative values, unknown keys, unknown pipeline id and `is_dev` mismatch are rejected at the boundary (unit and CLI `run --dry-run`, exit code 3). A YAML `!!python` tag is not executed (`safe_load`). |
| Malformed judge, client | `test_malformed_judge.py` | 16 passed, 5 xfailed | Real `instructor` against a fake Ollama server: a garbage reply makes exactly `max_retries + 1` requests, then raises; recovery on a later attempt works; schema-invalid replies raise; seed and temperature reach the wire. |
| Malformed judge, pipeline | `test_malformed_judge_pipeline.py` | 11 | 5-question run with scripted judge failures: a raising judge gives `judge_failed`; NaN rows are stored with a reason; aggregates ignore NaN; `run_validity` marks the run INCONCLUSIVE below 80 % measured. |
| Log redaction | `test_log_redaction.py` | 34 | `redact()` and `RedactingFilter` remove e-mails, `sk-`/`ghp_`/`hf_` keys, Bearer tokens and `key=value` secrets; run ids, hashes and passage ids stay readable. |
| Secrets exposure, logic | `test_secrets_check.py` | 43 | `scripts/check_secrets.py` on throw-away git repos: finds `.gitignore` gaps, bad `.env.example`, secret shapes in tracked files and in deleted-but-committed history; reports are masked. |
| Secrets exposure, guard | `test_secrets_exposure.py` | 3 | On the real repo: `.env*` ignored, `.env.example` placeholders only, no tracked secret shape. |

Manual check, not in the suite (grows with the repo): `python scripts/check_secrets.py`
(whole history, all branches). **Result Oct 26: PASS** on all five checks.

## 2. Mitigations in place

| Risk | Mitigation | Evidence |
|---|---|---|
| Invalid or hostile config | Pydantic v2, `extra="forbid"`, frozen models, caps (`top_k` 1..50, `chunk_size` 1..4000, `overlap` < `chunk_size`, `max_workers` 1..8, seeds >= 0), YAML `safe_load` | `test_config_validation.py` |
| Judge returns garbage | Bounded retries, then raise; failure recorded per question (`judge_failed`) | `test_malformed_judge*.py` |
| Judge output yields no usable score | Faithfulness returns NaN; the comparator treats it as unmeasured; `run_validity` makes the run INCONCLUSIVE below 80 % measured, so a bad judge cannot silently pass a comparison | `test_malformed_judge_pipeline.py` |
| Secret committed to git | `.gitignore` covers `.env`, `.env.*` (except `.env.example`); `check_secrets.py` plus 3 guard tests | `test_secrets_*.py`, manual scan PASS |

## 3. Known risks and accepted behaviour

| ID | Risk or behaviour | Status | Action |
|---|---|---|---|
| S1 | **Prompt injection.** Retrieved chunks are pasted into the generator prompt as plain text, with no delimiter and no "context is data" instruction. A passage can try to steer the answer. | **Not tested yet; known risk.** The test (B4) and the prompt fix are deferred together because any prompt change alters generator output and breaks reproduction of the canonical runs. | After Day 14: add `prompt_version`, harden the prompt (also fix the typo `"ou are a question-answering..."`), add `test_prompt_injection.py` as `xfail(strict)` first, with an opt-in test against the real model. |
| S2 | **Judge verdict not range-checked.** `NLIStatementOutput.verdict` is an unconstrained `int`; 2, -1 or 7 pass the schema and count as "supported", so a bad judge can inflate faithfulness. | Pinned as `xfail(strict)` (5 xfailed). No source change. | Change the field to `Literal[0, 1]` after Day 14; the xfails then flip to XPASS and the markers must be removed. |
| S3 | **NaN faithfulness row is tagged `pass`.** No claims, or a verdict-count mismatch, gives NaN without an exception; the row is stored `judged` and tagged `pass`. | Accepted and pinned. The comparator reads NaN as unmeasured, and `run_validity` protects the comparison. | Optional: tag such rows differently. |
| S4 | **Same-count mismatched verdicts.** A verdict list with the right length but statements in the wrong order or with different text is accepted. | Untested. | Match verdict statements to claims, or document as a judge-quality limit. |
| S5 | **Lax integer coercion.** `true`, `"5"` and `5.0` are accepted as ints in config; `5.5` and `"1e3"` are rejected. | Accepted (D11-Q7) and pinned. | None. |
| S6 | **YAML alias bomb** (nested anchors that expand exponentially). `safe_load` blocks code execution but not expansion. | **Untested.** Configs are written by the user. | Add a size cap on config files if configs ever come from untrusted sources. |
| S7 | **NaN reason is persisted** in `metric_results_json`. | Intended; relevant to open item D9-Q23 (faithfulness reason persistence). | Decide with D9-Q23. |

## 4. Log redaction: limits and status

`ragregress/core/logging.py` provides `redact()` and `RedactingFilter`.

**Status: unit-tested, not an active control.** Nothing in the harness imports it yet, so logs
are currently not redacted. Wiring is planned for Day 13, together with an integration test
that logs through the real handler.

| Limit | Detail |
|---|---|
| `print()` bypasses logging | `pipeline.py` and the generator still `print()` debug lines (questions, ids, `repr(exception)`). The filter cannot see them. Replace with `logging` on Day 13. |
| Attach to handlers | A filter on a logger does not see records from child loggers. Use `handler.addFilter(RedactingFilter())`. |
| Shared record | The filter rewrites the `LogRecord` in place, so every handler sees the redacted text. |
| `stack_info` not redacted | Only the message and exception text are. |
| Shapes only | E-mails, `sk-`/`ghp_`/`hf_` keys, Bearer tokens, `api_key`/`token`/`password`/`passwd`/`secret` pairs. Phone numbers, IP addresses and user paths are not redacted. Over-redaction is possible (`token=abc` is redacted by design). |

## 5. Secrets check: scope and limits

`scripts/check_secrets.py` (standard library plus `git`; exit 0 clean, 1 findings, 2 cannot run):

| Check | What it does |
|---|---|
| gitignore | `.env`, `.env.local`, `.env.production`, `.env.development` ignored; `.env.example` not ignored |
| env-example | placeholders only (no secret shapes, no real-looking values) |
| tracked | no tracked file has a secret shape; no tracked `.env*` except the example |
| env-history | no `.env*` (except the example) ever committed |
| history | no commit on any branch ever added a secret-shaped line |

Limits:
- Only tracked files and committed history are scanned; new, uncommitted files are not.
- Shapes only (`sk-`, `ghp_`-family, `hf_`, `AKIA`, Slack `xox`, private-key header). A secret with no recognisable shape outside a `.env` file is not found.
- A secret that was pushed must be **rotated**; removing it from git is not enough.
- Findings are masked (first 4 characters and length), so the report is safe to paste.
- Known quirk: `.gitignore` has `artifacts/` followed by `!artifacts/reports/.gitkeep`; git cannot re-include a file under an excluded folder, so the second line does nothing.
- The project has no real third-party secret by design (local Ollama; `api_key="ollama"` is a dummy value required by the OpenAI-compatible endpoint).

## 6. Follow-ups

| When | Item |
|---|---|
| Day 13 | Wire `RedactingFilter` into the logging setup; replace `print()` with `logging`; add an integration test through the real handler |
| After Day 14 | Prompt hardening and `prompt_version` (S1), then B4 injection test; `verdict: Literal[0, 1]` (S2); clear the `xfail` markers |
| Optional | Config file size cap (S6); verdict-to-claim matching (S4); distinct tag for NaN rows (S3) |
