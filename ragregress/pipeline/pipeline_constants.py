
from enum import Enum

RUN_STATUS = "run_status"
RUN_STATUS_DETAILS = "run_status_detail"
FAILURE_MODE = "failure_mode"
PER_QUERY_RESULT = "per_query_results"

QUERY_STATUS ="query_status"


QA_ID = "qa_id"
QUESTION = "question"
GOLD_IDS = "gold_ids"
GOLD_ANSWERS = "gold_answers"
CHUNKS = "chunks"
GENERATION = "generation"
METRIC_RESULTS = "metric_results"
RETRY_COUNT = "retry_count"
LAST_ERROR = "last_error"
LATENCY_MS = "latency_ms"
CATEGORY = "category"

class Node(str, Enum):
    # LangGraph node names.

    INGEST = "ingest"

    PROCESS_QUERY = "process_query"
    
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    JUDGE = "judge"
    FAILED_QUERY = "failed_query"
    FAILED_RUN = "failed_run"
    AGGREGATE = "aggregate"
    PERSIST = "persist"
    COMPARE = "compare"
    
    def __str__(self) -> str:
        return str(self.value)


class Route(str, Enum):
    """Routing-decision labels returned by conditional-edge functions that
    are NOT themselves node names (e.g. "retry" means "go back to the same
    node this branch is already in" -- it isn't a distinct node)."""
 
    FAN_OUT = "fan_out"
    RETRY = "retry"

    def __str__(self) -> str:
        return str(self.value)

class QueryStatus(str, Enum):
    """QueryState["status"] values 
    Set by a branch's own node

    """

    PENDING = "pending"
    RETRIEVED = "retrieved"
    RETRIEVE_FAILED = "retrieve_failed"
    GENERATED = "generated"
    GENERATE_FAILED = "generate_failed"
    JUDGED = "judged"
    FAILED = "failed"

    def __str__(self) -> str:
        return str(self.value)

class RunStatus(str, Enum):
    #RunState["status"] values, set by Ingest and read by _route_after_ingest.
 
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

    def __str__(self) -> str:
        return str(self.value)

class FailureMode(str, Enum):
 
    RETRIEVAL_MISS = "retrieval_miss"
    LOW_FAITHFULNESS = "low_faithfulness"
    LOW_CORRECTNESS = "low_correctness"
    PASS = "pass"
    FAILED = "failed"

    def __str__(self) -> str:
        return str(self.value)