"""Dataset Validator Node implementation."""

import os
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List
from utils.logger import get_logger

log = get_logger("dataset_validator")

def dataset_validator_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Inspects the cloned temporary directory storage paths and builds metadata."""
    log.start("Executing dataset validation and structure profiling")
    
    target_path_str = state.get("target_path", "").strip()
    if not target_path_str:
        log.error("Missing critical state property: target_path")
        return {
            "execution_success": False,
            "error_message": "Target path was not provided in the system state.",
            "file_count": 0,
            "all_files": []
        }
        
    target_path = Path(target_path_str)
    valid_extensions = ('.csv', '.xlsx', '.xls')
    valid_files: List[str] = []
    
    if target_path.is_file():
        if target_path.suffix.lower() in valid_extensions:
            valid_files.append(str(target_path.absolute()))
    elif target_path.is_dir():
        for file in target_path.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extensions:
                valid_files.append(str(file.absolute()))

    file_count = len(valid_files)
    log.info("Discovered %d structural data asset(s) inside temporary workspace environment", file_count)

    if file_count == 0:
        log.warn("No structured target tables found at path: %s", target_path_str)
        return {
            "execution_success": False,
            "error_message": f"No valid CSV or Excel files located at target: {target_path_str}",
            "file_count": 0,
            "all_files": []
        }

    metadata_summary: Dict[str, Any] = {}
    for file_str in valid_files:
        file_path = Path(file_str)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df_chunk = pd.read_excel(file_str, nrows=5)
            else:
                df_chunk = pd.read_csv(file_str, nrows=5)
            
            metadata_summary[file_path.name] = {
                "absolute_path": file_str,
                "file_size_mb": round(file_size_mb, 2),
                "file_type": file_path.suffix.lower().replace('.', ''),
                "columns": list(df_chunk.columns),
                "inferred_dtypes": {col: str(dtype) for col, dtype in df_chunk.dtypes.items()}
            }
            log.info("Profiled cloned file [%s] | Size: %.2f MB", file_path.name, file_size_mb)
            
        except Exception as e:
            log.error("Failed parsing structural metrics for temporary file %s: %s", file_path.name, str(e))
            return {
                "execution_success": False,
                "error_message": f"Corrupt file layout encountered: {file_path.name}.",
                "file_count": file_count,
                "all_files": valid_files
            }

    log.end("Structural target validations finalized cleanly")
    
    return {
        "all_files": valid_files,
        "file_count": file_count,
        "dataset_metadata": metadata_summary,
        "execution_success": True,
        "error_message": None
    }