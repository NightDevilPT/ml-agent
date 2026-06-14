"""Splitter Export Node.

Modifies the existing processed train_dataset.csv file in-place, slicing 20% 
of its records off to populate the test_dataset.csv file.
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

from workflow.state import MLState
from utils.logger import get_logger

log = get_logger("splitter_export")


def splitter_export_run(state: MLState) -> Dict[str, Any]:
    """Reads the existing processed train file, splits it 80/20, and updates the workspace."""
    log.section("In-Place Data Partition Engine Initiated")

    workspace_path_str = state.get("clone_workspace", "")
    
    if not workspace_path_str:
        log.error("Splitter Aborted: Missing 'clone_workspace' path variable inside global state.")
        return {"consolidation_feedback": "Splitter Error: Active workspace path is undefined."}

    processed_dir = Path(workspace_path_str) / "processed-datasets"
    train_file_path = processed_dir / "train_dataset.csv"
    test_file_path = processed_dir / "test_dataset.csv"

    if not train_file_path.exists():
        log.error("Splitter Aborted: Target processed file not found at: %s", train_file_path)
        return {"consolidation_feedback": f"Splitter Error: Base train file missing at {str(train_file_path)}"}

    try:
        log.info("Loading existing processed train dataset: %s", train_file_path.name)
        df_processed = pd.read_csv(train_file_path)
        
        # Guard check against empty or tiny datasets
        if len(df_processed) < 5:
            log.warning("Dataset row-count is too low to split (< 5 records). Creating a fallback test set.")
            df_train = df_processed.copy()
            df_test = df_processed.copy()
        else:
            log.info("Removing 20%% of rows from train file and allocating to test dataset...")
            # split rows using a fixed random state for reproducibility
            df_train, df_test = train_test_split(df_processed, test_size=0.2, random_state=42)

        log.info("Overwriting train_dataset.csv with the remaining 80%% of records...")
        df_train.to_csv(train_file_path, index=False)

        log.info("Saving the extracted 20%% of records to test_dataset.csv...")
        df_test.to_csv(test_file_path, index=False)

        log.info("In-place partition complete. Train: %d rows | Test: %d rows", len(df_train), len(df_test))

        # Update absolute string paths back to centralized state parameters
        return {
            "train_path": str(train_file_path.absolute()),
            "test_path": str(test_file_path.absolute())
        }

    except Exception as split_fault:
        log.error("Splitter Engine Fault: Extraction script failure: %s", str(split_fault))
        return {"consolidation_feedback": f"Splitter Core Exception Error Trace: {str(split_fault)}"}