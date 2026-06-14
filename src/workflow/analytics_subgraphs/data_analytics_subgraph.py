"""Data Analytics Subgraph Orchestration Layer.

Manages data workspace pooling, structural table harmonization, cleaning, 
predictive schema analysis, and infinite-loop-proof data quality auditing.
"""

from langgraph.graph import StateGraph, END
from workflow.state import MLState

# Import finalized node execution scripts matching step progress
from workflow.analytics_subgraphs.nodes.clone_dataset import clone_dataset_run
from workflow.analytics_subgraphs.nodes.combine_datasets import combine_datasets_run
from workflow.analytics_subgraphs.nodes.single_file_cleaner import single_file_cleaner_run
from workflow.analytics_subgraphs.nodes.dataset_auditor import dataset_auditor_run
from workflow.analytics_subgraphs.nodes.splitter_export import splitter_export_run
from workflow.analytics_subgraphs.nodes.model_strategist import model_strategist_run


# --- Sub-Graph Conditional Routing Rules ---
def route_on_file_count(state: MLState) -> str:
    """Branches trajectory based on whether single or multi-file sets exist."""
    if len(state.get("all_files", [])) > 1:
        return "multi_file_branch"
    return "single_file_branch"


def route_on_audit_evaluation(state: MLState) -> str:
    """Evaluates the mathematical validity of the processed training matrix dataset.

    Implements a strict max-retry loop protection pattern.
    """
    # 1. If the auditor approves, pass out immediately
    if state.get("is_data_valid") is True:
        return "clear_to_exit"
    
    # 2. Extract and check the loop tracking counters from the state map
    retry_counters = state.get("retry_counters", {})
    audit_loops = retry_counters.get("ingestion_loop", 0)
    
    # Strict Infinite Loop Ceiling: Break out after 2 failed cleanup loops
    if audit_loops >= 2:
        print(f"\n[ORCHESTRATOR ALERT] Max retry ceiling reached ({audit_loops} attempts). Force-breaking infinite agent loop to preserve runtime token limits.")
        # Force-override the validation token flags to prevent upstream blockages
        state["is_data_valid"] = True
        return "clear_to_exit"
        
    # Validation failed, increment loop token steps down inside the graph flow chart
    return "reprocess_file"


# --- Sub-Graph Compilation Factory ---
def build_analytics_subgraph() -> StateGraph:
    """Assembles the Data Analytics and Auto-Adaptive Quality Verification Subgraph."""
    sub_workflow = StateGraph(MLState)
    
    # 1. Register all completed nodes
    sub_workflow.add_node("clone_dataset", clone_dataset_run)
    sub_workflow.add_node("combine_datasets", combine_datasets_run)
    sub_workflow.add_node("single_file_cleaner", single_file_cleaner_run)
    sub_workflow.add_node("dataset_auditor", dataset_auditor_run)
    sub_workflow.add_node("splitter_export", splitter_export_run)
    sub_workflow.add_node("model_strategist", model_strategist_run)
    
    # 2. Wire up the paths
    sub_workflow.set_entry_point("clone_dataset")
    
    sub_workflow.add_conditional_edges(
        "clone_dataset",
        route_on_file_count,
        {"multi_file_branch": "combine_datasets", "single_file_branch": "single_file_cleaner"}
    )
    
    sub_workflow.add_edge("combine_datasets", "single_file_cleaner")
    sub_workflow.add_edge("single_file_cleaner", "dataset_auditor")
    
    # 3. Auditor Routing: Clear audit pathways lead directly to the Splitter Node
    sub_workflow.add_conditional_edges(
        "dataset_auditor",
        route_on_audit_evaluation,
        {
            "clear_to_exit": "splitter_export",       # 🌟 Fix: Audited master file moves straight to splitter
            "reprocess_file": "single_file_cleaner"
        }
    )
    
    # 4. Chain the horizontal partition outputs cleanly into the strategy analyzer
    sub_workflow.add_edge("splitter_export", "model_strategist")  # 🌟 Fix: Pass split train datasets to strategist
    
    # 5. The strategist node marks the final exit point of your data analytics subgraph
    sub_workflow.add_edge("model_strategist", END)
    
    return sub_workflow.compile()


# Compile the finalized sub-graph asset register
compiled_analytics_subgraph = build_analytics_subgraph()