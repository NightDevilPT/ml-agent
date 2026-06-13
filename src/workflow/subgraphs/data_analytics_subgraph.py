"""Data Analytics Subgraph Orchestration Layer.

Manages data workspace pooling, structural table harmonization, cleaning, 
and predictive schema analysis within an isolated tracking boundary.
"""

from langgraph.graph import StateGraph, END
from workflow.state import MLState

# Import ONLY your finalized clone node execution script
from workflow.subgraphs.nodes.clone_dataset import clone_dataset_run


def build_analytics_subgraph() -> StateGraph:
    sub_workflow = StateGraph(MLState)
    
    # Register the single node we have built
    sub_workflow.add_node("clone_dataset", clone_dataset_run)
    
    # Entry Point Execution Alignment
    sub_workflow.set_entry_point("clone_dataset")
    
    # Temporarily route directly to END to keep this subgraph runnable.
    # We will replace this line with 'add_conditional_edges' as next nodes are built.
    sub_workflow.add_edge("clone_dataset", END)
    
    return sub_workflow.compile()

# Compile the professional sub-graph asset runnable
compiled_analytics_subgraph = build_analytics_subgraph()