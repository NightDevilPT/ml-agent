"""LangGraph workflow definition for ML Agent."""

from langgraph.graph import StateGraph, END
from workflow.state import MLState
from workflow.nodes.clone_dataset import clone_dataset_run
from workflow.nodes.dataset_validator import dataset_validator_run
from workflow.nodes.analyze_algorithm import analyze_algorithm_run
from workflow.nodes.select_algorithm import select_algorithm_run
from workflow.nodes.scenario_generator import scenario_generator_run

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    
    # Create graph context
    workflow = StateGraph(MLState)
    
    # Register decoupled agent nodes matching the architectural workflow blueprint
    workflow.add_node("clone_dataset", clone_dataset_run)
    workflow.add_node("dataset_validator", dataset_validator_run)
    workflow.add_node("analyze_algorithm", analyze_algorithm_run)
    workflow.add_node("select_algorithm", select_algorithm_run)
    workflow.add_node("scenario_generator", scenario_generator_run)  # Registered Scenario Gen
    
    # Set entry checkpoint tracking point
    workflow.set_entry_point("clone_dataset")
    
    # Wire sequential pipeline logic dependencies cleanly
    workflow.add_edge("clone_dataset", "dataset_validator")
    workflow.add_edge("dataset_validator", "analyze_algorithm")
    workflow.add_edge("analyze_algorithm", "select_algorithm")
    workflow.add_edge("select_algorithm", "scenario_generator")    # User locked selection ➔ Scenarios generated
    workflow.add_edge("scenario_generator", END)                    # Finished execution lifecycle
    
    # Compile graph framework
    return workflow.compile()


# Singleton graph instance
app = build_graph()