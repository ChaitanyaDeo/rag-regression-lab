"""
ragregress/pipeline/graph.py
 
LangGraph Studio / `langgraph dev` entry point
 
Creates one concrete `Pipeline` instance (BuiltinRAGSystem with chunk size 500 / overlap 50,
an InstructorStructuredLLM judge, and the four metrics RecallAtK, MRR, TokenPRF1, Faithfulness),
then exposes its compiled graph as `graph`.
 
`langgraph.json` points to:
`ragregress/pipeline/graph.py:graph`
 
Use `max_workers=1` to match the production default

"""

from ragregress.rag.builtin_system import BuiltinRAGSystem
from ragregress.rag.chunker import ChunkerConfig
from ragregress.evaluation.structured_llm import InstructorStructuredLLM
from ragregress.evaluation.metrics import RecallAtK, MRR, TokenPRF1, Faithfulness
from ragregress.pipeline.pipeline import Pipeline

K = 5

JUDGE_MODEL = "qwen2.5:7b-instruct"

_rag_system = BuiltinRAGSystem(ChunkerConfig(chunk_size=500, overlap=50))

_judge_llm = InstructorStructuredLLM(JUDGE_MODEL, max_retries=3)

_metrics = [
    RecallAtK(k=K),
    MRR(k=K),
    TokenPRF1(),
    Faithfulness(llm=_judge_llm),
]

_pipeline = Pipeline(
    rag_system=_rag_system,
    metrics=_metrics,
    k=K,
    max_workers=1,  
)

#langgraph.json points at -- the compiled graph Studio visualizes and runs

graph = _pipeline.graph

