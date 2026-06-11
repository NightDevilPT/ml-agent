"""Clone Dataset Node implementation."""

import os
import uuid
import shutil
from pathlib import Path
from typing import Any, Dict, List
from utils.logger import get_logger

log = get_logger("clone_dataset")

def clone_dataset_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates an isolated temporary path structure and clones all raw target
    source files into it as the absolute first pipeline dependency guardrail.
    """
    log.start("Clone Dataset Node — Phase 1: Environment Provisioning & Data Duplication")
    
    target_path_str = state.get("target_path", "").strip()
    if not target_path_str:
        log.error("Missing critical state property: target_path")
        return {
            "execution_success": False,
            "error_message": "Target path was not provided to clone node."
        }
        
    target_path = Path(target_path_str)
    if not target_path.exists():
        log.error("Source path does not exist on host: %s", target_path_str)
        return {
            "execution_success": False,
            "error_message": f"Provided path does not exist: {target_path_str}"
        }

    # Gather source files from target input
    source_files: List[Path] = []
    if target_path.is_file():
        if target_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
            source_files.append(target_path)
    elif target_path.is_dir():
        for file in target_path.iterdir():
            if file.is_file() and file.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                source_files.append(file)

    if not source_files:
        log.warn("No structured tabular targets found at path: %s", target_path_str)
        return {
            "execution_success": False,
            "error_message": f"No valid files located to clone at target: {target_path_str}"
        }

    # Establish unique temporary environment layout matching user spec boundaries
    workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    cuid = uuid.uuid4().hex[:8]
    
    # clone_workspace targets the raw unique folder path base string
    clone_workspace_dir = workspace_root / ".temp" / f"ml_agent_{cuid}"
    cloned_data_target_dir = clone_workspace_dir / "datasets"
    
    log.info("Allocating isolated clone workspace framework at: %s", str(clone_workspace_dir))
    
    try:
        clone_workspace_dir.mkdir(parents=True, exist_ok=True)
        cloned_data_target_dir.mkdir(parents=True, exist_ok=True)
        
        new_cloned_file_paths: List[str] = []
        for source_file in source_files:
            destination_file_path = cloned_data_target_dir / source_file.name
            log.info("Cloning table asset: %s ➔ %s", source_file.name, destination_file_path.name)
            
            shutil.copy2(source_file, destination_file_path)
            new_cloned_file_paths.append(str(destination_file_path.absolute()))
            
    except Exception as io_error:
        log.error("Failed to execute data cloning operations: %s", str(io_error))
        return {
            "execution_success": False,
            "error_message": f"Cloning node encountered file-system block: {str(io_error)}"
        }
        
    log.end("Workspace initialized. Original source tables duplicated safely.")
    
    # Overwrite target_path to point to the temporary datasets location for downstream processing
    return {
        "clone_workspace": str(clone_workspace_dir.absolute()),
        "target_path": str(cloned_data_target_dir.absolute()), 
        "all_files": new_cloned_file_paths,
        "execution_success": True,
        "error_message": None
    }