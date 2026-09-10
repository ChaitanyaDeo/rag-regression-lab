# RAG Regression Lab

A configurable regression-testing harness for RAG pipelines — catches silent quality regressions when chunk size, embedding model, prompt, or top-k change, the way code has regression tests but RAG configs typically don't.

*(Full README structure — Problem / Solution / Architecture / Demo / Key engineering decisions / Evaluation methodology / Results / Failure cases / Observability / Security / Setup / API / Limitations / Future work — lands Day 14 per the project plan's Section 22. This is the Day 1 scaffold plus the target-metrics stub required before any real run exists.)*

## Target Metrics

Written down **before** any real pipeline run exists — per Section 26's anti-fabrication discipline: these are engineering-judgment expectations for a healthy config, not measured results. The eventual Day 9/14 report shows these against actual numbers, side by side, never backfilled.

| Metric | Target | Reasoning |
|---|---|---|
| Recall@5 | 0.80–0.85 | Small (2067-passage), topically distinct corpus; MiniLM embeddings generally perform well at this scale despite SQuAD's deliberate lexical variation between question and passage phrasing |
| MRR | 0.65–0.75 | Typically trails Recall@5 — being in the top 5 doesn't guarantee rank 1 |
| Answer F1 | 0.55–0.65 | A 3B instruct model doing free-form generation (not dedicated span-extraction) will underperform SQuAD's fine-tuned extractive baselines (~0.80–0.90), but should do reasonably with correct context in front of it |
| Faithfulness | 0.80–0.90 | Given retrieval mostly surfaces the correct gold passage, a well-behaved config should stay well-grounded — this metric mainly catches misconfiguration, not fundamental model limits |

## Dataset

Uses SQuAD (`rajpurkar/squad`), `validation` split only — not `rag-mini-wikipedia` as originally named in the project plan (see the Day 1 chat task list, DEC-06, for why: `rag-mini-wikipedia`'s QA config has no passage-link field, and its answers include non-extractive yes/no values that break the plan's documented fallback strategy). SQuAD's `context` field gives an exact-match gold passage per question.

- Corpus: 2067 unique passages (deduplicated from `context` values)
- `dev`: 15 questions, hand-tagged
- `val`: 50 questions, LLM-assisted category tagging (spot-checked)
- `test`: 200 questions,

All three sampled once with `EVAL_SEED=42`, never resampled.

## Category taxonomy

Defaults to CRAG's 8 types, but this project uses a restricted set — `simple`, `simple_with_condition`, `comparison` (see `configs/categories_squad.json`) — since SQuAD questions are single-passage, extractive, and guaranteed-answerable by construction, meaning `false_premise`, `multi-hop`, `aggregation`, `set`, and `post_processing_heavy` essentially never legitimately apply here. See DEC-11 in the chat task list for the full reasoning.