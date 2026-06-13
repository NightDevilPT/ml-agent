"""LangGraph workflow graph topology definition layout managing pipeline structures.

Includes structural circuit breakers to prevent infinite self-healing runtime loops.
"""

from langgraph.graph import StateGraph, END
from workflow.state import MLState

# Import the compiled data sub-graph macro block with updated naming convention
from workflow.subgraphs.data_analytics_subgraph import compiled_analytics_subgraph

# ================================================================
# Graph Builder Configuration Topology
# ================================================================
def build_graph() -> StateGraph:
    workflow = StateGraph(MLState)
    
    # Register the compiled analytics subgraph macro-node block
    workflow.add_node("data_analytics_subgraph_phase", compiled_analytics_subgraph)
    
    # Main Workflow Entry Point
    workflow.set_entry_point("data_analytics_subgraph_phase")
    
    # Set the subgraph to transition directly to END upon completion
    workflow.add_edge("data_analytics_subgraph_phase", END)
    
    return workflow.compile()

app = build_graph()