"""ML Architect Subgraph Orchestration Layer.

Manages automated script generation and disk serialization.
"""

from langgraph.graph import StateGraph, END
from workflow.state import MLState

# Clean folder-based functional node imports
from workflow.ml_subgraph.nodes.ml_script_architect import ml_script_architect_run
from workflow.ml_subgraph.nodes.script_io_writer import script_io_writer_run  # 🌟 Added


def build_ml_architect_subgraph() -> StateGraph:
    """Assembles the ML Code Architecture Subgraph using clean folder-based imports."""
    sub_workflow = StateGraph(MLState)
    
    # Register the nodes
    sub_workflow.add_node("ml_script_architect", ml_script_architect_run)
    sub_workflow.add_node("script_io_writer", script_io_writer_run)  # 🌟 Added
    
    # Establish sequential flow
    sub_workflow.set_entry_point("ml_script_architect")
    
    # 🌟 Chain the architect generation node straight into the IO disk writer node
    sub_workflow.add_edge("ml_script_architect", "script_io_writer")
    sub_workflow.add_edge("script_io_writer", END)
    
    return sub_workflow.compile()


# Compile the finalized sub-graph asset register safely
compiled_ml_architect_subgraph = build_ml_architect_subgraph()