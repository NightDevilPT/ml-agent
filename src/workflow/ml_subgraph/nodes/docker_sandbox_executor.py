"""Subprocess Docker Sandbox Execution Supervisor Node."""

import subprocess
from pathlib import Path
from typing import Dict, Any

from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("docker_sandbox_executor")


def docker_sandbox_executor_run(state: MLState) -> Dict[str, Any]:
    """Builds the container and triggers the chained training and evaluation sequence in one shot."""
    log.section("Initializing Chained Docker Sandbox Isolation Lifecycle")

    clone_workspace_str = state.get("clone_workspace", "").strip()
    if not clone_workspace_str:
        log.error("Executor Aborted: 'clone_workspace' path variable missing from state context.")
        return {
            "script_execution_success": False,
            "runtime_stderr": "Sandbox Runtime Failure: 'clone_workspace' path is completely empty."
        }

    workspace_root = Path(clone_workspace_str).resolve()
    
    # 🌟 DYNAMIC NAMING ALIGNMENT: Match the exact folder tag used by the code architect node
    workspace_folder_id = workspace_root.name  # Extracts 'ml_agent_bit-coin_xxxx' or matching dataset ID
    image_tag = f"{workspace_folder_id}:latest"
    container_name = f"{workspace_folder_id}-container"

    # Step 1: Compile the unique self-contained container image layers
    log.info(f"Compiling isolated container environment image layers [Tag: {image_tag}]...")
    build_cmd = ["docker", "build", "-t", image_tag, "."]
    build_proc = subprocess.run(
        build_cmd, cwd=workspace_root, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    
    if build_proc.returncode != 0:
        log.error("Docker build phase failed with status code: %d", build_proc.returncode)
        return {
            "script_execution_success": False,
            "runtime_stderr": f"[DOCKER IMAGE BUILD FAILURE]:\n{build_proc.stderr}"
        }

    # Step 2: Run the uniquely tagged container.
    log.info(f"Spawning core pipeline container process [Name: {container_name}]...")
    
    # Ensure any stale container instance using this exact name is scrubbed out first
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
    
    run_cmd = ["docker", "run", "--rm", "--name", container_name, image_tag]
    run_proc = subprocess.run(
        run_cmd, cwd=workspace_root, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    
    if run_proc.returncode != 0:
        log.error("Container execution sequence failed with status: %d", run_proc.returncode)
        return {
            "script_execution_success": False,
            "runtime_stderr": f"[CONTAINER RUNTIME FAILURE]:\n{run_proc.stderr}"
        }

    log.info("Chained training and holdout inference snapshot executed successfully.")
    return {
        "script_execution_success": True,
        "runtime_stdout": run_proc.stdout,
        "runtime_stderr": None,
        "model_prediction_accurate": None
    }