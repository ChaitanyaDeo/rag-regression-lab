from ragregress.evaluation.metrics.base import Metric, MetricResult
from ragregress.evaluation.metrics.retrieval_recall.metric import RecallAtK
from ragregress.evaluation.metrics.retrieval_mrr.metric import MRR
from ragregress.evaluation.metrics.token_prf1.metric import TokenPRF1
from ragregress.evaluation.metrics.faithfulness.metric import Faithfulness

# Explicit on purpose (not Metric.__subclasses__(): that depends on what happens to be imported).
# `ragregress compare` looks metrics up here by their stored name (DEC-D8-Q14).
# A new metric = import it above and add it to this tuple (plus __all__).
_ALL_METRICS = (RecallAtK, MRR, TokenPRF1, Faithfulness)
METRIC_REGISTRY: dict[str, type[Metric]] = {m.name: m for m in _ALL_METRICS}

if len(METRIC_REGISTRY) != len(_ALL_METRICS):
    raise RuntimeError("duplicate metric name in METRIC_REGISTRY")

__all__ = [
    "Metric", 
    "MetricResult", 
    "RecallAtK", 
    "MRR", 
    "TokenPRF1", 
    "Faithfulness",
    "METRIC_REGISTRY",
]