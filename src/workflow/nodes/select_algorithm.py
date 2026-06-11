"""Select Algorithm Node implementation."""

from typing import Any, Dict, List
from utils.logger import get_logger
from utils.hitl import ask_human

log = get_logger("select_algorithm")

def select_algorithm_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Presents recommendations to the user via HITL menu and records selection."""
    node_name = "select_algorithm"
    log.start("Select Algorithm Node — Phase 3B: Interactive Steering Gateway")
    
    recommendations = state.get("algorithm_recommendations", [])
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    
    if not recommendations:
        log.error("No modeling recommendations discovered in execution state.")
        return {
            "execution_success": False,
            "error_message": "Missing candidate recommendations array. Run analyze_algorithm first."
        }
        
    available_options: Dict[str, str] = {}
    mapping_helper: Dict[str, Dict[str, Any]] = {}
    
    for index, item in enumerate(recommendations, start=1):
        key = str(index)
        name = item.get("algorithm_name")
        package = item.get("package")
        weight = item.get("selection_score", 0)
        justification = item.get("justification", "No explicit justification provided.")
        is_best = " (RECOMMENDED BEST)" if item.get("is_primary_recommendation") else ""
        
        display_text = (
            f"[{package}] {name} | Suitability Score: {weight}/100{is_best}\n"
            f"        👉 Justification: {justification}\n"
        )
        available_options[key] = display_text
        mapping_helper[key] = item

    log.section("User Core Selection Menu")
    selected_key = ask_human(
        options=available_options,
        title="Model Pipeline Steering Input",
        description="Select a machine learning algorithm strategy based on your data snippet's properties.",
        default="1",
        style="warning"
    )
    
    chosen_algorithm_config = mapping_helper.get(selected_key)
    if not chosen_algorithm_config:
        log.error("Out-of-bounds menu mapping key captured: %s", selected_key)
        return {
            "execution_success": False,
            "error_message": "User selected an invalid response sequence."
        }
        
    log.info("User locked in strategy path selection: %s", chosen_algorithm_config.get("algorithm_name"))
    log.end("Interactive choice processing finalized cleanly")
    
    updated_node_tokens = {**historical_node_tokens, node_name: 0}
    return {
        "selected_algorithm_config": chosen_algorithm_config,
        "token_count": global_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True,
        "error_message": None
    }