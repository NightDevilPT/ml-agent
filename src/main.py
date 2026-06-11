"""Application entry point for the Agentic AutoML System."""

import os
import json
from pathlib import Path
from utils.logger import get_logger
from utils.hitl import ask_human
from workflow.graph import app

# Instantiate primary host framework terminal logger
log = get_logger("main_orchestrator")

def main():
    log.section("Initializing ML Agent Workflow")
    
    # Determine workspace root to structure the JSON output path cleanly
    workspace_root = Path(__file__).resolve().parent.parent
    output_json_path = workspace_root / "state_record.json"
    
    # --- HITL Dataset Selection Gate ---
    log.section("Human-In-The-Loop Data Selection")
    
    # Present a single option to direct the user to input the absolute path
    path_prompt_option = {"1": "Enter a custom absolute directory path"}
    
    # Trigger the interactive HITL control panel
    ask_human(
        options=path_prompt_option,
        title="Dataset Location Selector",
        description="Please provide the absolute path to your dataset folder below.",
        style="info"
    )
    
    # Direct absolute path text entry
    print("\n" + "="*60)
    custom_path_str = input("Enter the absolute path to your target data collection directory: ").strip()
    print("="*60 + "\n")
    
    # Guardrail: If no path is passed, return and halt execution immediately
    if not custom_path_str:
        log.warn("No path was provided. Aborting workflow execution.")
        return
        
    chosen_dataset_path = Path(custom_path_str)
    log.info("Target data pathway established via HITL input: %s", str(chosen_dataset_path))

    # 1. Build initial centralized state payload 
    initial_state = {
        "target_path": str(chosen_dataset_path),
        "clone_workspace": "",  # This will be set by the workflow
        "dataset_metadata": None,
        "all_files": [],
        "file_count": 0,
        "execution_success": True,
        "error_message": None,
        "token_count": 0
    }
    
    log.info("System Stateful Context successfully primed. Invoking LangGraph runtime.")
    
    # 2. Invoke the compiled graph engine app
    final_state = None
    try:
        final_state = app.invoke(initial_state)
        
        # 3. Process structural output execution states
        log.section("Workflow Lifecycle Evaluation")
        if final_state.get("execution_success"):
            log.info("Pipeline execution completed successfully.")
            log.info("Discovered file count metrics: %d", final_state.get("file_count", 0))
            log.info("Identified file keys: %s", list(final_state.get("dataset_metadata", {}).keys()))
        else:
            log.error("Pipeline terminal fault caught: %s", final_state.get("error_message"))
            
    except Exception as e:
        log.error("Fatal unhandled runtime exception encountered outside sandbox: %s", str(e))
        # Fallback payload structure to log if an absolute framework crash occurs
        final_state = initial_state.copy()
        final_state["execution_success"] = False
        final_state["error_message"] = f"Fatal system error: {str(e)}"

    # --- JSON State Persistence Layer ---
    if final_state:
        log.section("State Serialization Layer")
        try:
            # Get clone_workspace from final_state
            clone_workspace = final_state.get("clone_workspace")
            
            if clone_workspace:
                # Save JSON inside clone_workspace folder
                clone_workspace_path = Path(clone_workspace)
                clone_workspace_path.mkdir(parents=True, exist_ok=True)
                output_json_path = clone_workspace_path / "state_record.json"
                log.info("Serializing terminal context records into JSON file at: %s", output_json_path)
            else:
                # Fallback to root if clone_workspace not found
                log.warn("clone_workspace not found in final_state, saving to root directory")
                output_json_path = Path(__file__).resolve().parent.parent / "state_record.json"
            
            # Serialize the state dictionary with clean formatting rules
            with open(output_json_path, "w", encoding="utf-8") as json_file:
                json.dump(final_state, json_file, indent=4, ensure_ascii=False)
                
            log.info("State history successfully updated at: %s", str(output_json_path))
        except Exception as serialize_error:
            log.error("Failed to commit tracking state records to disk: %s", str(serialize_error))


if __name__ == "__main__":
    main()