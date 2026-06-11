"""Sandbox Executor Node — Phase 7: Isolated Code Execution & Validation."""

from pathlib import Path
from typing import Any, Dict
import docker
from utils.logger import get_logger

log = get_logger("sandbox_executor")

def sandbox_executor_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mounts the cloned workspace into the prepared Docker container and executes
    the training loop script via uv while trapping standard error logs.
    """
    node_name = "sandbox_executor"
    log.start("Sandbox Executor Node — Phase 7: Launching Sandboxed Execution")
    
    clone_workspace_str = state.get("clone_workspace", "")
    image_tag = state.get("sandbox_image_tag", "")
    historical_node_tokens = state.get("node_tokens", {})
    
    if not clone_workspace_str or not image_tag:
        log.error("Missing workspace path trajectories or built Docker image references.")
        return {"execution_success": False, "error_message": "Missing prepper dependencies."}
        
    workspace_path = Path(clone_workspace_str)
    
    try:
        client = docker.from_env()
        log.info("Spawning container from target image layer: %s", image_tag)
        
        # Run the training script via 'uv run train.py' in the container.
        # Live-mount the host workspace directory directly to /app inside Docker
        container_output = client.containers.run(
            image=image_tag,
            command="uv run train.py",
            volumes={
                str(workspace_path.absolute()): {
                    'bind': '/app',
                    'mode': 'rw'
                }
            },
            working_dir="/app",
            stderr=True,
            stdout=True,
            remove=True  # Automatically clean up container tracking spaces on completion
        )
        
        log.info("Container execution output received successfully.")
        decoded_logs = container_output.decode("utf-8")
        print("\n--- SANDBOX CONTAINER CONSOLE OUTPUT ---\n", decoded_logs, "\n----------------------------------------\n")
        
    except docker.errors.ContainerError as exc:
        # Script execution encountered a breakdown (Non-zero exit code caught!)
        error_logs = exc.stderr.decode("utf-8")
        log.warn("Sandbox execution breakdown detected! Script failed to terminate cleanly.")
        print("\n--- CRASH LOG IDENTIFIED ---\n", error_logs, "\n----------------------------\n")
        
        updated_node_tokens = {**historical_node_tokens, node_name: 0}
        return {
            "execution_success": False,
            "error_message": error_logs,  # Capture raw error logs to pass directly back to ml_architect
            "node_tokens": updated_node_tokens
        }
    except Exception as general_err:
        log.error("Host environment container bridge broken: %s", str(general_err))
        return {"execution_success": False, "error_message": str(general_err)}
        
    log.end("Sandboxed model pipeline completed successfully without any execution faults!")
    updated_node_tokens = {**historical_node_tokens, node_name: 0}
    
    return {
        "execution_success": True,
        "error_message": None,
        "node_tokens": updated_node_tokens
    }