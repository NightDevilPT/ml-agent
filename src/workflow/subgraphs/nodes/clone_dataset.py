"""Workspace Provisioner Node.

Creates structural storage partitions inside the temporary sandboxed folder.
"""

import uuid
import shutil
from pathlib import Path
from typing import Dict, Any
from utils.logger import get_logger
from workflow.state import MLState

# Instantiate dedicated logger for ingestion scope tracking
log = get_logger("clone_dataset")

def clone_dataset_run(state: MLState) -> Dict[str, Any]:
    """Parses host pathway inputs, provisions workspace directories, and stages raw files."""
    log.section("Initializing Isolated Datasets Directory Workspace")
    
    # Extract source targets safely from incoming state envelope
    target_path_str = state.get("target_path", "").strip()
    if not target_path_str:
        log.error("Target execution aborted: 'target_path' missing from framework context state.")
        return {
            "is_data_valid": False,
            "consolidation_feedback": "Error: Core parameter 'target_path' is missing or unpopulated."
        }
        
    source_path = Path(target_path_str)
    if not source_path.exists():
        log.error("Directory path matching input targeting signature does not exist: %s", target_path_str)
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"Error: Target data collection directory path does not exist: {target_path_str}"
        }

    # Discover and filter raw dataset sheets matching platform processing specs
    raw_files = []
    valid_extensions = {'.csv', '.xlsx', '.xls'}
    
    if source_path.is_file():
        if source_path.suffix.lower() in valid_extensions:
            raw_files.append(source_path)
    elif source_path.is_dir():
        raw_files = [
            f for f in source_path.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

    if not raw_files:
        log.error("Target folder validation broke: No structured tabular data arrays (.csv, .xlsx, .xls) found.")
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"Error: No valid spreadsheets found inside route location: {target_path_str}"
        }

    # Generate workspace folder paths: .temp/ml_agent_datasetfolderName_{cuid}
    src_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    cuid = uuid.uuid4().hex[:8]
    folder_prefix = source_path.stem if source_path.is_file() else source_path.name
    
    # Secure structural string clean up for cross-platform folder safety
    safe_folder_prefix = "".join(c_char for c_char in folder_prefix if c_char.isalnum() or c_char in ('_', '-'))
    clone_workspace_dir = src_root / ".temp" / f"ml_agent_{safe_folder_prefix}_{cuid}"
    
    datasets_staging_dir = clone_workspace_dir / "datasets"
    processed_output_dir = clone_workspace_dir / "processed-datasets"

    try:
        # Generate physical folder boundaries on host system disk
        datasets_staging_dir.mkdir(parents=True, exist_ok=True)
        processed_output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Workspace environment boundaries generated inside storage workspace: %s", clone_workspace_dir)
        
        # Clone raw spreadsheet metrics into local isolation folder
        cloned_paths_list = []
        for file_to_clone in raw_files:
            destination_target_file = datasets_staging_dir / file_to_clone.name
            shutil.copy2(file_to_clone, destination_target_file)
            cloned_paths_list.append(str(destination_target_file.absolute()))
            log.info("Spreadsheet tracked and staged: %s -> %s", file_to_clone.name, destination_target_file.name)
            
    except Exception as io_error:
        log.error("Fatal file system IO blockage hit while constructing pipeline structures: %s", str(io_error))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"File System Workspace failure: Unable to write to disk. Trace: {str(io_error)}"
        }

    log.info("Data clone setup phase concluded. Total files locked into workspace scope: %d", len(cloned_paths_list))
    
    # Return dictionary slice updating state changes seamlessly
    return {
        "clone_workspace": str(clone_workspace_dir.absolute()),
        "all_files": cloned_paths_list,
        "is_data_valid": False,
        "consolidation_feedback": None,
        "retry_counters": {"ingestion_loop": 0},
        "token_count": state.get("token_count", 0),
        "node_tokens": state.get("node_tokens", {})
    }