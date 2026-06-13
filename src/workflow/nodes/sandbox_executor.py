"""
Sandbox Executor Node — Phase 6: Isolated Environment Compilation Run
Runs the written train.py code inside a sandboxed container.
"""

from pathlib import Path
from typing import Any, Dict
from utils.logger import get_logger
from utils.sandbox import MlSandbox

log = get_logger("sandbox_executor")

def sandbox_executor_run(state: Dict[str, Any]) -> Dict[str, Any]:
    node_name = "sandbox_executor"
    
    # 🌟 HALTING GUARDRAIL: Stop execution immediately if upstream architect failed
    if state.get("execution_success") is False:
        return {
            "execution_success": False,
            "error_message": state.get("error_message", "Halted downstream executor: Upstream script architect collapsed.")
        }
        
    log.start("Sandbox Executor Node — Phase 6: Executing Code Inside Container Sandbox")
    
    clone_workspace = state.get("clone_workspace", "")
    processed_dir = Path(clone_workspace) / "processed_datasets"
    processed_files = [str(f.absolute()) for f in processed_dir.iterdir() if f.is_file() and f.suffix.lower() == '.csv'] if processed_dir.exists() else []
    
    target_script = Path(clone_workspace) / "train.py"
    if not target_script.exists():
        log.error("Target training script train.py missing from workspace root.")
        return {"execution_success": False, "error_message": "train.py code missing from workspace root."}

    with open(target_script, "r", encoding="utf-8") as f:
        code_string = f.read()

    combined_packages = ["pandas", "joblib", "scikit-learn", "xgboost"]
    
    with MlSandbox(memory_limit="8g", timeout=400) as sandbox:
        preparation_success = sandbox.prepare(
            dataset_files=processed_files,
            code_string=code_string,
            packages=combined_packages
        )
        
        if not preparation_success:
            return {"execution_success": False, "error_message": "Failed copying workspace assets to container mounts."}
            
        execution_result = sandbox.execute()
        
        if not execution_result.success:
            log.warn("Container training compilation crash caught! Redirecting back into self-healing loops...")
            return {
                "latest_sandbox_logs": execution_result.logs + f"\n{execution_result.error_message}",
                "execution_success": True  # Keep graph alive for routing transitions
            }
            
        extracted_model_path = sandbox.extract_model()
        if extracted_model_path:
            import shutil
            shutil.copy2(extracted_model_path, Path(clone_workspace) / "model.joblib")
            log.info("Model binary extracted to workspace safely.")
        else:
            return {
                "latest_sandbox_logs": "Execution returned exit code 0, but failed to output a model.joblib artifact file matrix.",
                "execution_success": True
            }

    log.end("Sandbox run successfully validated. Moving to behavioral assertion verification pipelines.")
    return {
        "latest_sandbox_logs": "",
        "execution_success": True,
        "error_message": None
    }