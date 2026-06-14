"""LangGraph workflow graph topology definition layout managing pipeline structures.

Includes structural circuit breakers to prevent infinite self-healing runtime loops.
"""

from langgraph.graph import StateGraph, END
from workflow.state import MLState

# 🌟 Updated import path to use standard python underscores matching your renamed folder
from workflow.analytics_subgraphs.data_analytics_subgraph import compiled_analytics_subgraph
from workflow.ml_subgraph.ml_architect_subgraph import compiled_ml_architect_subgraph


# ================================================================
# Graph Builder Configuration Topology
# ================================================================
def build_graph() -> StateGraph:
    """Assembles and compiles the top-level parent ML pipeline orchestration graph."""
    workflow = StateGraph(MLState)
    
    # 1. Register the compiled subgraph macro-nodes
    workflow.add_node("data_analytics_subgraph_phase", compiled_analytics_subgraph)
    workflow.add_node("ml_architect_subgraph_phase", compiled_ml_architect_subgraph)
    
    # 2. Set the Main Workflow Entry Point
    workflow.set_entry_point("data_analytics_subgraph_phase")
    
    # 3. Establish the Phase Handoff Sequence Connection
    workflow.add_edge("data_analytics_subgraph_phase", "ml_architect_subgraph_phase")
    
    # 4. Terminal Phase Exit
    workflow.add_edge("ml_architect_subgraph_phase", END)
    
    return workflow.compile()


# Compile the global application state machine tracking engine
app = build_graph()