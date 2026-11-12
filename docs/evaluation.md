# Evaluation methodology

What RAG Regression Lab measures, how a comparison becomes a verdict, and how to read the results. Architecture: [architecture.md](architecture.md). Usage: [README](../README.md).

## Contents

1. [Dataset](#1-dataset)
2. [Metrics](#2-metrics)
3. [Faithfulness in detail](#3-faithfulness-in-detail)
4. [Per-run aggregation](#4-per-run-aggregation)
5. [Failure modes](#5-failure-modes)
6. [Comparison and verdict](#6-comparison-and-verdict)
7. [Diagnostics](#7-diagnostics)
8. [Targets](#8-targets)
9. [Results](#9-results)
10. [Reproducibility](#10-reproducibility)
11. [How to read the results](#11-how-to-read-the-results)
12. [Ragas cross-check](#12-ragas-cross-check)

## 1. Dataset

| Item | Value |
|---|---|
| Source | SQuAD v1.1, `validation` split (`rajpurkar/squad`) |
| Corpus | every distinct `context` value, deduplicated: 2,067 passages, ids `p0000000`… in first-seen order (`data/passages.json`) |
| Gold | each question links to exactly one gold passage (`gold_passage_ids`) and keeps all SQuAD answer strings |
| Splits | `dev` 15, `val` 50, `test` 200 questions, each sampled once with seed 42 (`data/splits/*.json`) |
| Categories | restricted set `simple`, `simple_with_condition`, `comparison` (hand-tagged on dev, LLM-assisted with spot checks on val/test) |
| Build | `python -m ragregress.benchmark.loader`; the data is committed, so runs do not depend on downloads |

- The splits are three independent samples of the same question set and may overlap: they are fixed views at different speeds (dev for smoke tests, val for comparisons), not a train/test split.
- `dev` runs are marked `is_dev` and cannot be compared (the comparator rejects them).
- Why SQuAD: an exact gold passage per question makes retrieval metrics unambiguous. The price is an easy retrieval task (see [§11](#11-how-to-read-the-results)).

## 2. Metrics

Eleven metrics are available, all computed per question; each declares its regression gate (or none) on its class. Which ones run is chosen per run (`ragregress run --metric a,b` or `all`, the API's `metrics` list; `ragregress metrics` and `GET /metrics` list them). With no choice, the four **default** metrics run: Recall@k, MRR, Token F1 and Faithfulness. The metric choice is not part of `config_hash`. The results in [§9](#9-results) use the default four.

| Metric | Name | Stage | Definition | Default | Gate |
|---|---|---|---|---|---|
| Recall@k | `recall_at_k` | retrieval | `|gold ∩ top-k retrieved passage ids| / |gold|`; with one gold passage this is 1 if the gold is in the top k, else 0 | yes | 0.05 |
| MRR | `mrr` | retrieval | `1 / rank` of the first gold passage within the top k, else 0 | yes | 0.05 |
| Precision@k | `precision_at_k` | retrieval | distinct gold passages in the top k, divided by k (with one gold passage the maximum is 1/k) | no | none |
| nDCG@k | `ndcg_at_k` | retrieval | binary relevance, duplicates counted once (first occurrence), ideal ranking uses `min(|gold|, k)` | no | none |
| Context recall | `context_recall` | retrieval | share of the reference answer's sentences that the retrieved context supports (judge, one call; [§3](#context-recall-and-context-precision)) | no | none |
| Context precision | `context_precision` | retrieval | average precision of the retrieved chunks judged useful for the reference answer (judge, one call per chunk; [§3](#context-recall-and-context-precision)) | no | none |
| Token F1 | `token_prf1` | generation | SQuAD token F1 against each gold answer, best one kept | yes | 0.05 |
| Exact match | `exact_match` | generation | 1 if the normalised answer equals any gold answer (SQuAD normalisation), else 0 | no | none |
| ROUGE-L | `rouge_l` | generation | ROUGE-L F-measure (`rouge-score`, no stemming) against each gold answer, best one kept | no | none |
| BLEU | `bleu` | generation | sentence BLEU (`sacrebleu`, effective order, lower-cased), scaled to 0–1, against all gold answers | no | none |
| Faithfulness | `faithfulness` | generation | supported claims / extracted claims, judged against the retrieved context | yes | 0.05 |

Details:
- **k** is the config's `top_k` (5 in all shipped configs). Only the first k retrieved chunks are scored, whatever the pipeline returns.
- **Retrieved ids** are the chunks' `source_passage_id`, so several chunks of the gold passage count once.
- **Token F1 normalisation** (SQuAD style): lower-case, remove punctuation, remove the articles *a / an / the*, collapse whitespace, split on whitespace. If either side normalises to no tokens, F1 is 1 when both are empty, else 0.
- **Answers are citation-stripped** before scoring: the built-in generator writes `[source: p…]` markers, which are removed from `answer`.
- **Gate** = the largest tolerated worsening of the paired mean, in the metric's own units (0.05 = five points on a 0–1 scale). All metrics are higher-is-better. A metric with no gate is reported and compared but never decides the verdict; thresholds are set only after the baseline is re-run with the metric.
- **Redundancy on SQuAD:** with one gold passage per question, Precision@k is Recall@k divided by k and nDCG@k tracks MRR; Exact match is expected near 0 for a sentence-style generator. They matter on datasets with several gold passages.
- **ROUGE-L** keeps articles ("The Eiffel Tower" vs "Eiffel Tower" scores 0.8), unlike Token F1's normalisation. **BLEU** is low and noisy for short gold answers.

## 3. Faithfulness in detail

Ragas-style, implemented in the project (`ragregress/evaluation/metrics/faithfulness/`):

```mermaid
flowchart LR
  A[question + answer] --> S[1. statement extraction<br/>judge LLM -> list of atomic statements]
  S -->|no statements| N1[NaN: no claims extracted]
  S --> V[2. verification<br/>judge LLM: statements + retrieved context -> verdict 0/1 + reason each]
  V -->|verdict count != statement count| N2[NaN: count mismatch]
  V --> F[3. score = supported / total]
```

| Setting | Value |
|---|---|
| Judge | `qwen2.5:7b-instruct` via Ollama by default (configurable: `judge_provider`, `judge_model`) |
| Determinism | temperature 0, seed `judge_seed` (42) |
| Structured output | `instructor` JSON mode against Pydantic schemas (`StatementGeneratorOutput`, `NLIStatementOutput`) |
| Retries | up to 3 retries per LLM call on schema-invalid output (4 requests); then the metric raises |
| Context | the retrieved chunks' text, each prefixed with its passage id |
| Missing inputs | empty question, answer or context raises (the metric is unmeasured for that question) |

Outcomes for one question:

| Situation | Stored value | Effect |
|---|---|---|
| normal | 0.0–1.0 | measured |
| no claims extracted, or verdict count mismatch | NaN with a reason | dropped from means; unmeasured for validity |
| judge output invalid after retries, or missing input | metric absent, `last_error` set, tag `judge_failed` | unmeasured; lowers validity |

Faithfulness measures **groundedness, not correctness**: an answer can be fully supported by the retrieved text and still wrong (see the example in [§9](#failure-cases)).

### Context recall and context precision

Two opt-in, judge-based retrieval metrics, following Ragas (`ragregress/evaluation/metrics/context_recall/`, `context_precision/`). Both judge the retrieved chunks against a **reference**: the question's gold answers as one text (`ragregress/evaluation/reference.py`: distinct after normalisation, empty ones dropped, one sentence per line). A question with no non-empty gold answer gives NaN with a reason and no judge call. Neither reads the generated answer; they ask whether the retrieval served the question.

| | Context recall | Context precision |
|---|---|---|
| Question answered | does the retrieved text contain what the reference states? | are the useful chunks ranked first? |
| Judge calls | 1 per question: the judge splits the reference into sentences and marks each 1 (supported by the context) or 0 | 1 per retrieved chunk (`top_k` per question): is this chunk useful for arriving at the reference? 1 or 0 |
| Score | supported sentences / sentences | average precision: for each useful chunk at rank r, precision@r; summed and divided by the number of useful chunks; 0 if none |
| Differs from Recall@k / Precision@k | judges the content, not the passage id: catches a gold passage whose retrieved chunk lacks the answer, and an answer found in another passage | judges usefulness of each chunk and rewards putting the useful ones first |
| Structured output | verdict `attributed` 0/1 per sentence | verdict 0/1 per chunk |
| Failure | invalid judge output after retries, or a missing input, raises | one failing chunk call raises; the question is lost for this metric |

- The judge is the configured one (same as Faithfulness, same determinism). Judged metrics lengthen a run in proportion to their calls; `context_precision` is the costlier (on the 15-question dev split with the default judge, about 18 s of judge time per question against 12 s for `context_recall`).
- On SQuAD the reference is a short span, so each sentence is a bare answer; `context_recall` then behaves like a content-aware "is the answer in the retrieved text" check.
- The `context_precision` judge prompt was tightened after inspection: chunks about the same topic but lacking the answer must score 0. It was checked on a handful of dev questions only (see [§11](#11-how-to-read-the-results)).
## 4. Per-run aggregation

| Aggregate | Rule |
|---|---|
| Metric value | mean over `judged` rows that have the metric; NaN values dropped |
| Latency | p50 and p95 per stage (`retrieve`, `generate`, `judge`) and for the per-question total; nearest-rank percentile; informational, never gated |
| Run status | `healthy` if the retrieval health probe passed, else `unhealthy` (no questions run) |

The run-level means are what `ragregress run` prints. Comparisons do **not** use them; they recompute paired means over common rows ([§6](#6-comparison-and-verdict)).

## 5. Failure modes

Each question gets one tag (first match wins, in this order):

| Tag | Condition |
|---|---|
| `failed` | retrieval or generation failed after 3 attempts, or every metric crashed |
| `judge_failed` | judged, but at least one configured metric is missing (it crashed) |
| `retrieval_miss` | Recall@k = 0 |
| `low_faithfulness` | faithfulness < 0.5 |
| `low_correctness` | token F1 < 0.3 |
| `pass` | none of the above |

Tags are derived from the same metrics they describe, so grouping by failure mode is partly circular: it helps locate regressions, it does not explain them.

The tags use `recall_at_k`, `faithfulness` and `token_prf1`; a rule whose metric was not selected is skipped. If none of the three is selected, every judged row is `pass` (a run of only `context_recall`, for example).

## 6. Comparison and verdict

### Row classes

| Class | When | Treated as |
|---|---|---|
| measured | row judged | its values (a missing metric is unmeasured for that metric only) |
| system failure | row failed with no answer (retrieve / generate exhausted retries) | 0 on every metric, **counts as measured** |
| evaluator failure | row failed but has an answer (judging crashed) | unmeasured on every metric |

A system failure is the pipeline's fault and should hurt its scores; an evaluator failure is the judge's fault and should not decide anything.

### Comparability guards

| Condition | Result |
|---|---|
| unknown run id, a dev run, missing database | error, exit 8 |
| different `eval_split` | error, exit 8 |
| different `top_k` | warning; `recall_at_k` ungated |
| different judge (provider + model) | warning; `faithfulness` ungated |
| a metric computed in only one of the runs | shown as n/a; not gated |
| legacy run (no `pipeline` in its stored config) | warning |
| different question sets | warning; only common rows are compared |

### Paired statistics

For each metric: the rows measured in both runs (`n_common`), the mean of each run over those rows, `delta = after − before`, and per-row counts of better / worse / tie (tie = |difference| ≤ 1e-9).

### Validity

`validity = rows with every gated metric measured / rows attempted`, per run. Below **80 %** in either run, the comparison cannot be trusted.

### Verdict

```mermaid
flowchart TD
  A[inputs comparable?] -->|no| E8[error, exit 8]
  A -->|yes| B{no gated metric,<br/>validity < 80 % in either run,<br/>or a gated metric with no common rows?}
  B -->|yes| I[INCONCLUSIVE, exit 7]
  B -->|no| C{any gated metric worse<br/>than its threshold?}
  C -->|yes| F[FAIL, exit 6]
  C -->|no| P[PASS, exit 0]
```

- "Worse than its threshold" means `before − after > threshold + 1e-9` (direction-aware). A worsening of exactly the threshold passes.
- An INCONCLUSIVE verdict also lists any metric that would have failed, marked as not trusted.
- The verdict is a regression gate: it says whether a change made a gated metric worse by more than tolerated, not which config is better.

## 7. Diagnostics

Read-only breakdown of one comparison (`ragregress compare --worst N`, the HTML report, the API):

| Output | Rule |
|---|---|
| Worst rows per gated metric | rows that got worse, sorted by worsening (then qa_id), top N |
| Report drill-downs | rows ranked by their largest worsening across gated metrics, then by how many metrics got worse; shows both answers, scores, retrieved passages and missing gold passages |
| Groups | by category and by the candidate's failure-mode tag: mean delta and worse / better counts per gated metric; groups with fewer than 3 rows are flagged |
| Notes | system failures in baseline / candidate are marked on the row |

Diagnostics never change the verdict or the exit code.

## 8. Targets

Written down before any real run (`configs/targets/squad_val.yaml`, from the README's DEC-03) and never adjusted afterwards:

| Metric | Target | Reasoning at the time |
|---|---|---|
| Recall@5 | 0.80–0.85 | small, topically distinct corpus; MiniLM should do well despite SQuAD's paraphrasing |
| MRR | 0.65–0.75 | trails recall: in the top 5 is not rank 1 |
| Answer F1 | 0.55–0.65 | a 3B generative model with the right context, below fine-tuned extractive models |
| Faithfulness | 0.80–0.90 | with the gold passage usually retrieved, answers should stay grounded |

The report shows each run against these ranges (below / within / above), using the same common-row means as the metric table.

## 9. Results

val split, 50 questions, top-k 5, generator `qwen2.5:3b-instruct`, judge `qwen2.5:7b-instruct`, default metrics. The opt-in metrics of [§2](#2-metrics) are implemented and tested but no val-split results are reported for them here. Each candidate changes one thing from `baseline` (MiniLM-L6, chunk 500 / overlap 50). Re-checked against the database on Nov 2, 2026.

| Metric | Target | baseline | chunk_1000 | bge_small | rerank_minilm |
|---|---|---|---|---|---|
| Change | | — | chunk 1000 | `BAAI/bge-small-en-v1.5` | cross-encoder, top 20 → 5 |
| Validity | | 50/50 | 48/50 | 49/50 | 48/50 |
| Recall@5 | 0.80–0.85 | 0.8800 | 0.9200 (+0.0400; 3 b / 1 w) | 0.8800 (0.0000; 4 b / 4 w) | 0.9400 (+0.0600; 4 b / 1 w) |
| MRR | 0.65–0.75 | 0.7157 | 0.6890 (−0.0267; 9 b / 10 w) | 0.8100 (+0.0943; 13 b / 5 w) | 0.8867 (+0.1710; 15 b / 5 w) |
| Token F1 | 0.55–0.65 | 0.2132 | 0.2530 (+0.0398; 14 b / 7 w) | 0.2430 (+0.0297; 13 b / 12 w) | 0.2454 (+0.0322; 15 b / 8 w) |
| Faithfulness | 0.80–0.90 | 0.7391 | 0.7766 (+0.0375; 11 b / 7 w) | 0.7296 (−0.0148; 11 b / 10 w) | 0.7170 (−0.0259; 8 b / 8 w) |
| Verdict | | — | PASS | PASS | PASS |

b / w = questions better / worse than the baseline. Values are paired means over common rows; because the common rows differ per pair, the baseline's faithfulness is 0.7391 / 0.7444 / 0.7429 in the three comparisons (shown: standalone, n = 48).

**Target vs actual:** Recall@5 above target for all; MRR within (baseline) or above (bge_small, rerank_minilm); Token F1 far below for all; faithfulness below for all.

### Failure cases

- **Reranker pushes the gold passage out** (`5725cb33271a42140099d1df`, "What does ctenophore use to capture prey?", gold *colloblasts*): baseline has the gold at rank 3 and mentions colloblasts (Recall 1, F1 0.1176, faithfulness 1.0); rerank_minilm loses the gold from the top 5 and gives a generic answer (Recall 0, F1 0, faithfulness 1.0). Grounded, but wrong.
- **Judge output that does not fit the schema** (`572a07a86aef0514001551d5`, rerank run): the verification reply lost its `verdict` field on all 4 attempts; faithfulness unmeasured, tag `judge_failed`.
- **Judge inconsistency on refusals:** identical answers and identical refusals scored 1.0 in one run and 0.0 in another; in the rerank run 9 of 10 refusals scored 0.0.

## 10. Reproducibility

| Run | Environment | Result |
|---|---|---|
| `val-demo-rerank-20261028-133845` | Windows host, GPU, CLI | reference |
| `val-2106599c-20261102-102626` | Linux container, CPU-only torch, `POST /runs` | same `config_hash`; **identical value on every question for all four metrics** (0 better / 0 worse) |

Only latency differed: retrieve p50 77 ms (GPU) vs 466 ms (CPU); total p50 about 23 s per question in both, dominated by the judge (two LLM calls per question). With temperature 0 and fixed seeds, a stored run is a reproducible baseline across machines and over time.

## 11. How to read the results

| Caveat | Consequence |
|---|---|
| **n = 50** | little statistical power; a 0.04 recall change is 2 questions. Read deltas as directional; the gate is for catching regressions, not for ranking configs. |
| **Judge consistency** | the 7B judge scores the same answer differently across contexts; refusals are often scored 0. Small faithfulness deltas (such as rerank's −0.0259, 8 better / 8 worse) are within judge noise. |
| **Token F1 vs a sentence-style generator** | SQuAD gold answers are short spans; a generator that answers in sentences scores low even when correct (likely cause of F1 far below target; not measured separately). |
| **Easy retrieval** | one gold passage, a small, topically distinct corpus: recall saturates quickly; no hybrid retrieval was tested. |
| **Failure-mode groups** | tags come from the same metrics, so groups describe rather than explain. |
| **Category groups** | val has three categories of very different sizes; small groups are flagged and anecdotal. |
| **Faithfulness ≠ correctness** | read it together with token F1 (see the ctenophore case). |
| **Judged context metrics** | `context_recall` and `context_precision` depend on the same 7B judge: on a few inspected dev questions the verdicts were right, but a first prompt marked same-topic chunks as useful, and quality was checked on few rows. Treat their values as indicative until validated on more rows or a stronger judge. |
| **Duplicate passages in the top k** | the retriever can return several chunks of one passage (three of five on some dev questions), so Recall@k counts it once while Precision@k, nDCG@k and `context_precision` see each chunk. |
| **Opt-in metrics are ungated** | they cannot fail a run; compare their numbers by eye until thresholds are set from a re-run baseline. |

## 12. Ragas cross-check

`scripts/ragas_faithfulness_check.py` scores a handful of questions with both this project's faithfulness metric and Ragas's, using the same local judge model, as a sanity check of the implementation (Ragas is not used anywhere else). The comparison is stored in `results/ragas_faithfulness_check.json`. It needs `pip install ragas==0.4.3 langchain-community==0.3.31` and expects Ollama at `localhost:11434`.