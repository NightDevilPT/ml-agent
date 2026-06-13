"""Data Analytics Subgraph Orchestration Layer.

Manages data workspace pooling, structural table harmonization, cleaning, 
and predictive schema analysis within an isolated tracking boundary.
"""

from langgraph.graph import StateGraph, END
from workflow.state import MLState

# Import finalized node execution scripts matching step progress
from workflow.subgraphs.nodes.clone_dataset import clone_dataset_run
from workflow.subgraphs.nodes.combine_datasets import combine_datasets_run
from workflow.subgraphs.nodes.single_file_cleaner import single_file_cleaner_run

# --- Sub-Graph Conditional Routing Rules ---
def route_on_file_count(state: MLState) -> str:
    """Branches trajectory based on whether single or multi-file sets exist."""
    if len(state.get("all_files", [])) > 1:
        return "multi_file_branch"
    return "single_file_branch"

def build_analytics_subgraph() -> StateGraph:
    sub_workflow = StateGraph(MLState)
    
    # Register all completed processing nodes
    sub_workflow.add_node("clone_dataset", clone_dataset_run)
    sub_workflow.add_node("combine_datasets", combine_datasets_run)
    sub_workflow.add_node("single_file_cleaner", single_file_cleaner_run)
    
    # Entry Point Execution Alignment
    sub_workflow.set_entry_point("clone_dataset")
    
    # Route dynamically based on raw file count dimensions
    sub_workflow.add_conditional_edges(
        "clone_dataset",
        route_on_file_count,
        {
            "multi_file_branch": "combine_datasets",
            "single_file_branch": "single_file_cleaner"
        }
    )
    
    # Both node branches converge directly into END for this verification step
    sub_workflow.add_edge("combine_datasets", END)
    sub_workflow.add_edge("single_file_cleaner", END)
    
    return sub_workflow.compile()

# Compile the sub-graph asset
compiled_analytics_subgraph = build_analytics_subgraph()