

"""
Top level evaluation entry point

Runs the RASystem over a test dataset once per row, 
socres every requested Metric against the same row data.

No metric calls rag_system itself; 
no pipeline stage runs unless something in `metrics` needs it.

User only needs to implement RAGSystem and pick which Metric instance to run.
-- they never implement StructuredLLM unless they want a fully
custom, non-LangChain judge.
"""

from typing import Dict, List, Set
from tqdm import tqdm

from ragregress.benchmark.schemas import QAPair
from ragregress.rag.interface import RAGSystem
from ragregress.evaluation.metrics.base import Metric

GENERATION_FIELDS = {'answer', 'context'}

def evaluate(rag_system: RAGSystem, qa_pairs: List[QAPair], 
             metrics: List[Metric], k:int = 5)->Dict:

    if not rag_system:
        raise ValueError("Provide a RAGSystem")

    if not qa_pairs or not isinstance(qa_pairs, list) or any(not isinstance(qa_pair, QAPair) for qa_pair in qa_pairs):
        raise ValueError("Provide a list of QAPair to evaluate")

    if not isinstance(metrics, (type(None), list)):
        raise TypeError(
            "Metrics should be provided in a list, e.g: metrics=[MRR()]"
        )

    if isinstance(metrics, list) and any(not isinstance(m, Metric) for m in metrics):
        raise TypeError(
            "All metrics must be initialised metric objects, e.g: metrics=[BleuScore(), AspectCritic()]"
        )

    if metrics is None:
        from ragregress.evaluation.metrics import Faithfulness
        from ragregress.evaluation.metrics import RecallAtK
        from ragregress.evaluation.metrics import MRR
        from ragregress.evaluation.metrics import TokenPRF1

        metrics = [TokenPRF1, MRR, RecallAtK, Faithfulness]

    needed = set().union(*(m.required_inputs for m in metrics) if metrics else set())

    needs_generation = bool(needed & GENERATION_FIELDS)

    rows: List[Dict] = []

    for qa in tqdm(qa_pairs, desc="Evaluating QA pairs", unit="qa"):
        chunks = rag_system.retrieve(qa.question, k=k)
        retrieved_ids = [c.source_passage_id for c in chunks]

        row_data = {
            "retrieved_ids": retrieved_ids,
            "gold_ids": qa.gold_passage_ids,
            "question": qa.question,
            "gold_answers": qa.answer,
        }

        if needs_generation:
            result = rag_system.generate(qa.question, chunks)

            row_data["answer"] = result.answer
            row_data["context"] = chunks

        row_result = {"qa_id": qa.qa_id}

        for metric in metrics:
            kwargs = {field: row_data[field] for field in metric.required_inputs}
            row_result[metric.name] = metric.score(**kwargs)

        rows.append(row_result)


    aggregates = {}

    for m in metrics:
        values =  [row[m.name].value for row in rows]

        finite_values = [v for v in values if v == v]  # drop NaN

        aggregates[m.name] = (sum(finite_values)/ len(finite_values) if finite_values else float("nan"))

    return {"rows":rows, "aggregates": aggregates}


