# Test fixtures

| File | Contents |
|---|---|
| `passages_5doc.json` | 5 passages (`Passage` shape), copied verbatim from `data/passages.json`. The "3-doc" fixture = the first 3. |
| `qa_5doc.json` | 5 QA pairs (`QAPair` shape), copied verbatim from `data/splits/val.json`, one per passage, same order. |

Real passage ids and qa_ids are kept so every row traces back to the corpus. Validated by `tests/unit/test_fixtures.py`.

Source: SQuAD v1.1 (Rajpurkar et al., 2016), licensed CC BY-SA 4.0.
