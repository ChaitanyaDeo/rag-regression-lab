"""
Retrieval-stage metrics: Recall@k and MRR, scored against each QA pair's
ground-truth gold_passage_id (exact id match against data/passages.json's
id space — never an LLM judgment). Deterministic: same retrieved ids and
gold id always produce the same score

Only Recall@k and MRR are implemented. 

Precision@k, Hit Rate@k, and nDCG@k
are deliberately deferred, not rejected: SQuAD gives exactly one relevant
passage per question, which makes Precision@k a constant rescaling of
Recall@k (Precision@k = Recall@k / k), Hit Rate@k identical to Recall@k,
and nDCG@k track the same rank signal MRR already gives. They'd earn
their place the moment a pipeline/dataset has multiple relevant passages
per query -- revisit then
"""

from typing import Dict, List, Set
from ragregress.benchmark.schemas import QAPair
from ragregress.rag.interface import RAGSystem

from ragregress.evaluation.metrics import RecallAtK
from ragregress.evaluation.metrics import MRR



def evaluate_retrieval(qa_pairs: List[QAPair], rag_system: RAGSystem, k:int)->Dict:
    """
    Runs rag_system.retrieve() for every QA pair, calculate score per query against gold_passage_id
    and returns both the per-query rows (for manual spot-checking) and the aggregate means.

    Built on top of recall_at_k/mrr rather than duplicating their logic, so those two stay 
    independently unit-testable

    """

    rows: List[Dict] = []

    for qa in qa_pairs:
        gold_ids = qa.gold_passage_ids
        gold_set = set(gold_ids)


        chunks = rag_system.retrieve(qa.question, k = k)
        retrieved_ids = [c.source_passage_id for c in chunks]
        top_k_ids = retrieved_ids[:k]

        rank = None

        for idx, rid in enumerate(top_k_ids):
            if rid in gold_set:
                rank = idx + 1
                break

        rows.append(
            {
                "qa_id": qa.qa_id,
                "question": qa.question,
                "gold_passage_ids": qa.gold_passage_ids,
                "retrieved_ids": retrieved_ids,
                "rank": rank,
                "recall_at_k": RecallAtK(k = k).score(retrieved_ids=retrieved_ids, gold_ids=gold_ids),
                "mrr": MRR(k=k).score(retrieved_ids=retrieved_ids, gold_ids=gold_ids)
            }
        )

    n = len(rows)
    mean_recall = sum(row["recall_at_k"].value for row in rows) / n if n else 0.0
    mean_mrr = sum(row["mrr"].value for row in rows) / n if n else 0.0


    return {
        "k": k,
        "n_queries": n,
        "mean_recall_at_k": mean_recall,
        "mean_mrr": mean_mrr,
        "rows": rows,
    }


