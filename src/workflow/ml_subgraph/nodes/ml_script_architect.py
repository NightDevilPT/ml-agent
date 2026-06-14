"""ML Script Architect Node.

Handles structured LLM orchestration to generate Windows-compliant, constrained ML training code assets.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from pydantic import BaseModel, Field

from workflow.state import MLState
from utils.llm import extract_token_usage, get_llm
from utils.logger import get_logger

log = get_logger("ml_script_architect")


class ArchitectScriptReport(BaseModel):
    rationale: str = Field(
        description="Brief professional architectural explanation of the generated python training pipeline script."
    )
    raw_python_code: str = Field(
        description="The complete, standalone executable Python script code containing data splitting, model instantiation, training, and model serialization (.joblib)."
    )


def ml_script_architect_run(state: MLState) -> Dict[str, Any]:
    """Generates a cross-platform, Windows-optimized Python model training script based on your chosen algorithm."""
    log.section("ML Script Generation Engine Initiated")

    train_path_str = state.get("train_path", "")
    chosen_target = state.get("chosen_target", "")
    chosen_algorithm = state.get("chosen_algorithm", "")
    workspace_path_str = state.get("clone_workspace", "")
    
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})

    if not train_path_str or not chosen_target or not chosen_algorithm or not workspace_path_str:
        log.error("Architect Aborted: State vector missing required parameters.")
        return {
            "consolidation_feedback": "Architect Error: Prerequisite configuration markers are empty.",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "ml_script_architect": 0}
        }

    train_file_path = Path(train_path_str)

    try:
        df_train = pd.read_csv(train_file_path)
        train_snippet = df_train.head(3).to_csv(index=False, sep="|")
        available_features = list(df_train.columns)
        
        column_dtypes = {col: str(df_train[col].dtype) for col in df_train.columns}
        unique_value_counts = {col: int(df_train[col].nunique()) for col in df_train.columns}
    except Exception as io_err:
        log.error("Architect IO Error: Failed reading dataset snapshots from disk channels: %s", str(io_err))
        return {
            "consolidation_feedback": f"Architect IO Exception: {str(io_err)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "ml_script_architect": 0}
        }

    llm = get_llm(provider="gemini", temperature=0.0)
    structured_architect = llm.with_structured_output(ArchitectScriptReport, include_raw=True)

    # 🌟 CRITICAL CHANGE: System prompt explicitly commands Windows path handling using pathlib
    prompt = f"""
    You are an Elite Staff Machine Learning Infrastructure Automation Engineer working on a Windows environment. 
    Your task is to write a standalone, production-ready Python script that trains an ML model and saves it to disk.

    [TARGET PARAMETERS]
    - Target Feature(s) to predict (y): {str(chosen_target)}
    - Target Model Algorithm to build: {str(chosen_algorithm)}
    - Source Clean Training Data Path: {str(train_file_path.absolute())}

    [AVAILABLE DATASET CHANNELS]
    Columns List: {str(available_features)}
    Column Types: {json.dumps(column_dtypes)}
    Unique Counts: {json.dumps(unique_value_counts)}
    
    Data Snippet Preview:
    {train_snippet}

    [STRICT PROJECT ENVIRONMENT CONSTRAINTS]
    The script will execute inside an environment that ONLY has these dependencies installed. 
    You MUST ONLY use these exact libraries:
    - numpy>=2.4.6
    - pandas>=3.0.3
    - scikit-learn>=1.9.0
    - xgboost>=3.2.0
    - joblib>=1.5.3
    Do NOT import or use lightgbm, tensorflow, keras, pytorch, matplotlib, or seaborn.

    [🚨 CRITICAL WINDOWS FILE PATH REQUIREMENTS]
    - The host system runs Windows. You MUST use the `pathlib.Path` library to handle file paths dynamically.
    - Do NOT hardcode file path strings using forward slashes `/`. 
    - Always wrap files paths using `Path(r"...")` to safely format backslashes `\\` without escape character bugs.

    SCRIPT REQUIREMENT INSTRUCTIONS:
    1. Import `from pathlib import Path` at the top of the file.
    2. Read the clean data file into memory safely using `pd.read_csv(Path(r"{str(train_file_path.absolute())}"))`.
    3. Isolate the target column(s) from the feature spaces (X).
    4. Split the data matrix natively into Train/Test pairs using scikit-learn's train_test_split.
    5. Initialize the exact requested modeling algorithm: '{chosen_algorithm}'. Handle multi-output setups automatically if a list of targets was chosen.
    6. Execute .fit() on the training subsets. 
    7. Serialize the final fitted model using `joblib.dump(model, Path("trained_model.joblib"))` in the current working directory.
    
    Populate the ArchitectScriptReport configuration contract perfectly. Return ONLY the python code inside raw_python_code.
    """

    log.info("Emitting Windows-compliant code generation request to LLM.")
    try:
        response = structured_architect.invoke(prompt)
        report: ArchitectScriptReport = response["parsed"]
        node_spent = extract_token_usage(response["raw"])
        log.info("Automated python script generation step completed successfully. Tokens: %d", node_spent)
    except Exception as ai_fault:
        log.error("Platform Fault: LLM script architect failed to emit structured blueprint: %s", str(ai_fault))
        return {
            "consolidation_feedback": f"LLM Architect Exception: {str(ai_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "ml_script_architect": 0}
        }

    return {
        "generated_code_rationale": report.rationale,
        "generated_code_script": report.raw_python_code,
        "token_count": global_token_count + node_spent,
        "node_tokens": {**historical_node_tokens, "ml_script_architect": node_spent}
    }