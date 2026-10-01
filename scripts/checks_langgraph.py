from typing import TypedDict, Annotated, List, Union
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class ChildState(TypedDict, total=False):
    y: str
    shared: Annotated[List[int], operator.add]

def child_node(state):
    return {"shared": [1]}

child_builder = StateGraph(ChildState)
child_builder.add_node("child_node", child_node)
child_builder.add_edge(START, "child_node")
child_builder.add_edge("child_node", END)
child_graph = child_builder.compile()

class ParentState(TypedDict, total=False):
    x: str
    items: List[str]
    status: str
    shared: Annotated[List[int], operator.add]
    aggregated: dict

def ingest(state):
    print("ingest saw:", list(state.keys()))
    return {"status": "healthy"}

def route(state):
    if state.get("status") != "healthy":
        return "failed"
    return [Send("child", {"y": i}) for i in state["items"]]

def aggregate(state):
    print("aggregate saw:", list(state.keys()))
    return {"aggregated": {}}

def persist(state):
    print("persist saw:", list(state.keys()))
    return {}

def compare(state):
    return {}

def failed(state):
    return {"aggregated": {}}

g = StateGraph(ParentState)
g.add_node("ingest", ingest)
g.add_node("child", child_graph)
g.add_node("failed", failed)
g.add_node("aggregate", aggregate)
g.add_node("persist", persist)
g.add_node("compare", compare)

g.add_edge(START, "ingest")
g.add_conditional_edges("ingest", route, {"child": "child", "failed": "failed"})
g.add_edge("child", "aggregate")
g.add_edge("failed", "persist")
g.add_edge("aggregate", "persist")
g.add_edge("persist", "compare")
g.add_edge("compare", END)

graph = g.compile()
graph.invoke({"x": "hello", "items": ["a", "b"], "status": None, "shared": [], "aggregated": {}})