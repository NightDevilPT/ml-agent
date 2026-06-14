"""ML Architect Subgraph Orchestration Layer.

Manages automated script generation, disk serialization, containerized sandbox execution,
and linearized validation steps for precise single-pass execution.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from workflow.state import MLState

# Import production node modules
from workflow.ml_subgraph.nodes.ml_script_architect import ml_script_architect_run
from workflow.ml_subgraph.nodes.script_io_writer import script_io_writer_run
from workflow.ml_subgraph.nodes.docker_sandbox_executor import docker_sandbox_executor_run
from workflow.ml_subgraph.nodes.llm_prediction_validator import llm_prediction_validator_run


# --- Linear Phase Routing Controllers ---
def route_from_executor(state: MLState) -> str:
    """Evaluates container exit codes to determine if evaluation can proceed."""
    if state.get("script_execution_success") is not True:
        print("\n[ROUTER CRITICAL] Container runtime failed. Halting pipeline to preserve token budget.")
        return "halt_pipeline"
        
    print("\n[ROUTER CHECKPOINT] Subprocess code ran successfully. Forwarding stdout matrix to AI validation gate.")
    return "run_semantic_audit"


def route_from_validator(state: MLState) -> str:
    """Evaluates the semantic quality audit results and terminates the phase context."""
    if state.get("model_prediction_accurate") is True:
        print("\n[ROUTER SUCCESS] Sandbox execution and model predictions are completely verified. Exiting Subgraph.")
    else:
        print("\n[ROUTER WARNING] Semantic anomalies detected in output, but looping is disabled. Exiting gracefully.")
        
    return "exit_phase"


# --- Sub-Graph Compilation Factory ---
def build_ml_architect_subgraph() -> StateGraph:
    """Assembles the ML Code Architecture and Sandbox Execution Subgraph without looping overhead."""
    sub_workflow = StateGraph(MLState)
    
    # 1. Register operational execution nodes (Stripped code_self_healer node to save tokens)
    sub_workflow.add_node("ml_script_architect", ml_script_architect_run)
    sub_workflow.add_node("script_io_writer", script_io_writer_run)
    sub_workflow.add_node("docker_sandbox_executor", docker_sandbox_executor_run)
    sub_workflow.add_node("llm_prediction_validator", llm_prediction_validator_run)
    
    # 2. Establish Entry Point Boundary
    sub_workflow.set_entry_point("ml_script_architect")
    
    # 3. Static Sequential Execution Links
    sub_workflow.add_edge("ml_script_architect", "script_io_writer")
    sub_workflow.add_edge("script_io_writer", "docker_sandbox_executor")
    
    # 4. Separate One-Way Routing Logic for Executor Node
    sub_workflow.add_conditional_edges(
        "docker_sandbox_executor",
        route_from_executor,
        {
            "run_semantic_audit": "llm_prediction_validator",
            "halt_pipeline": END
        }
    )
    
    # 5. Separate One-Way Routing Logic for Validator Node (Wired to hit END directly)
    sub_workflow.add_conditional_edges(
        "llm_prediction_validator",
        route_from_validator,
        {
            "exit_phase": END
        }
    )
    
    return sub_workflow.compile()


# Compile the finalized sub-graph asset register safely
compiled_ml_architect_subgraph = build_ml_architect_subgraph()