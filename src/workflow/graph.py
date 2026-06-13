"""
LangGraph workflow graph topology definition layout managing pipeline structures.
Includes structural circuit breakers to prevent infinite self-healing runtime loops.
"""

from langgraph.graph import StateGraph, END
from workflow.state import MLState

# Local node imports
from workflow.nodes.clone_dataset import clone_dataset_run
from workflow.nodes.dataset_validator import dataset_validator_run
from workflow.nodes.data_preprocessor import data_preprocessor_run
from workflow.nodes.analyze_algorithm import analyze_algorithm_run
from workflow.nodes.select_algorithm import select_algorithm_run
from workflow.nodes.ml_architect import ml_architect_run
from workflow.nodes.sandbox_executor import sandbox_executor_run
from workflow.nodes.scenario_test_runner import scenario_test_runner_run
from workflow.nodes.generate_eval_script import generate_eval_script_run

# ================================================================
# Routing Condition Functions
# ================================================================
def evaluate_sandbox_compilation(state: MLState) -> str:
    """Checks if the container model build crashed at compile runtime."""
    # Hard circuit breaker: If upstream node explicitly returned an execution failure, exit immediately
    if state.get("execution_success") is False:
        return "halt_pipeline_failure"
        
    if state.get("latest_sandbox_logs"):
        return "recode_healing_loop"
    return "advance_to_testing"

def evaluate_test_assertions(state: MLState) -> str:
    """Checks if the generated predictions clear validation holdout specs."""
    # Hard circuit breaker: If upstream node explicitly returned an execution failure, exit immediately
    if state.get("execution_success") is False:
        return "halt_pipeline_failure"
        
    if state.get("test_failure_report"):
        return "recode_healing_loop"
    return "generate_artifacts"


# ================================================================
# Graph Builder Configuration Topology
# ================================================================
def build_graph() -> StateGraph:
    workflow = StateGraph(MLState)
    
    # 1. Register operational execution step run methods
    workflow.add_node("clone_dataset", clone_dataset_run)
    workflow.add_node("dataset_validator", dataset_validator_run)
    workflow.add_node("data_preprocessor", data_preprocessor_run)
    workflow.add_node("analyze_algorithm", analyze_algorithm_run)
    workflow.add_node("select_algorithm", select_algorithm_run)
    workflow.add_node("ml_architect", ml_architect_run)
    workflow.add_node("sandbox_executor", sandbox_executor_run)
    workflow.add_node("scenario_test_runner", scenario_test_runner_run)
    workflow.add_node("generate_eval_script", generate_eval_script_run)
    
    # 2. Enforce initialization target
    workflow.set_entry_point("clone_dataset")
    
    # 3. Direct solid immutable structural linkages
    workflow.add_edge("clone_dataset", "dataset_validator")
    workflow.add_edge("dataset_validator", "data_preprocessor")
    workflow.add_edge("data_preprocessor", "analyze_algorithm")
    workflow.add_edge("analyze_algorithm", "select_algorithm")
    workflow.add_edge("select_algorithm", "ml_architect")
    workflow.add_edge("ml_architect", "sandbox_executor")
    
    # 4. Sandbox routing evaluation path with structural crash guardrails
    workflow.add_conditional_edges(
        "sandbox_executor",
        evaluate_sandbox_compilation,
        {
            "recode_healing_loop": "ml_architect",
            "advance_to_testing": "scenario_test_runner",
            "halt_pipeline_failure": END  # Intercepts failures and halts iteration loop
        }
    )
    
    # 5. Holdout verification testing evaluation path with structural crash guardrails
    workflow.add_conditional_edges(
        "scenario_test_runner",
        evaluate_test_assertions,
        {
            "recode_healing_loop": "ml_architect",
            "generate_artifacts": "generate_eval_script",
            "halt_pipeline_failure": END  # Intercepts failures and halts iteration loop
        }
    )
    
    # 6. Final link: Complete graph once deployment script generation finishes cleanly
    workflow.add_edge("generate_eval_script", END)
    
    return workflow.compile()

app = build_graph()