"""AI Code Generation Engine Node."""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from pydantic import BaseModel, Field

from utils.llm import get_llm, extract_token_usage
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("ml_script_architect")


class MlArchitectOutput(BaseModel):
    required_pip_packages: list[str] = Field(description="Pip packages required.")
    training_script_code: str = Field(description="Raw Python training and evaluation unified code base.")
    workspace_readme_text: str = Field(description="Markdown format user manual text string.")


def ml_script_architect_run(state: MLState) -> Dict[str, Any]:
    """Generates optimized unified model execution scripts via Structured LLM."""
    log.section("Invoking ML Script Architect Engine")
    
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    
    clone_workspace = state.get("clone_workspace", "")
    train_path_str = state.get("train_path", "")
    test_path_str = state.get("test_path", "")
    
    chosen_algorithm = state.get("chosen_algorithm", "XGBoostRegressor")
    chosen_target = state.get("chosen_target", [])
    if isinstance(chosen_target, str):
        chosen_target = [chosen_target]
        
    problem_type = state.get("problem_type")
    if not problem_type:
        algo_lower = chosen_algorithm.lower()
        if "classifier" in algo_lower or "classification" in algo_lower or "logistic" in algo_lower or "svc" in algo_lower:
            problem_type = "Classification"
        else:
            problem_type = "Regression"
            
    consolidation_feedback = state.get("consolidation_feedback") or ""

    if not clone_workspace or not train_path_str or not test_path_str:
        return {"script_execution_success": False, "runtime_stderr": "Missing required data tracking paths."}

    workspace_folder_id = Path(clone_workspace).name
    train_folder_name = Path(train_path_str).parent.name

    try:
        df_train_glimpse = pd.read_csv(Path(train_path_str))
        train_features_columns = list(df_train_glimpse.columns)
        dataset_snapshot_string = df_train_glimpse.head(3).to_csv(index=False, sep="|")
    except Exception as read_fault:
        log.error("Architect IO Error: Failed profiling base training matrix schema: %s", str(read_fault))
        return {
            "script_execution_success": False,
            "runtime_stderr": f"IO Matrix Profile Failure: {str(read_fault)}"
        }

    llm = get_llm(provider="gemini", temperature=0.0)
    structured_llm = llm.with_structured_output(MlArchitectOutput, include_raw=True)
    
    input_features_only = [col for col in train_features_columns if col not in chosen_target]
    
    visible_features = [c for c in ["Open", "High", "Low", "Volume"] if c in input_features_only]
    if not visible_features:
        visible_features = input_features_only[:3]

    b3 = "\\`\\`\\`"

    prompt = f"""You are an expert ML Engineer. Generate an execution-ready unified Python script matching this configuration.

[PIPELINE SPECS]
- TASK STRATEGY: {problem_type}
- FRAMEWORK: {chosen_algorithm}
- TARGET VARIABLES (y): {json.dumps(chosen_target)}
- UNIQUE RUN ID: {workspace_folder_id}

[DATA MATRIX SCHEMA]
Columns: {json.dumps(train_features_columns)}
Snapshot:
{dataset_snapshot_string}
{consolidation_feedback}

[CODE SYSTEM INSTRUCTIONS]
1. Include `import argparse`, `import pandas as pd`, `import joblib`, `from pathlib import Path`, and necessary evaluation metrics from `sklearn.metrics`.
2. Framework Selection & Imports:
   - For {chosen_algorithm}, import both Classifier and Regressor variants (e.g., if XGBoost, import `XGBClassifier` and `XGBRegressor`; if RandomForest, import `RandomForestClassifier` and `RandomForestRegressor` from `sklearn.ensemble`; etc.).
3. Paths:
   BASE_DIR = Path(__file__).resolve().parent
   train_path = BASE_DIR / "{train_folder_name}" / "train_dataset.csv"
   test_path = BASE_DIR / "{train_folder_name}" / "test_dataset.csv"
   model_path = BASE_DIR / "model.joblib"
4. Execution Logic:
   - `--mode train`: 
     - Load `train_path` and preprocess features by dropping all target variables {json.dumps(chosen_target)} to get `X_train`.
     - For each target column `t` in {json.dumps(chosen_target)}:
       - Determine task type for `t` (Classification if `dtype` is non-numeric, bool, or if `df[t].nunique() <= 15`; otherwise Regression).
       - Instantiate the correct classifier or regressor model for {chosen_algorithm}.
       - Fit the model on `X_train` and `df[t]`.
       - Store the model in a dictionary: `models[t] = model`.
     - Save the `models` dictionary using `joblib.dump(models, model_path)`.
   - `--mode evaluate`:
     - Load `model_path`, load the ENTIRE test dataset from `test_path`. Preprocess features `X_test` by dropping target columns.
     - For each target column `t` in {json.dumps(chosen_target)}:
       - Predict using the model `models[t]` (use `predict()` for regression or classification).
       - Add the predictions to the evaluation dataframe under the column name `f"Pred_{{t}}"`.
       - Calculate appropriate metrics:
         - Classification metrics: accuracy_score, precision_score (weighted), recall_score (weighted), f1_score (weighted).
         - Regression metrics: mean_squared_error, mean_absolute_error, r2_score.
     - Print to stdout exactly in this block format:
       ```
       === START OF MODEL PERFORMANCE SCORECARD ===
       === MODEL PERFORMANCE SCORECARD ===
       Task Strategy: {problem_type}
       Selected Framework: {chosen_algorithm}
       Target Variables: {json.dumps(chosen_target)}
       
       <For each target in {json.dumps(chosen_target)}:
       Target: <target_name> (<Task_Type>)
       <Printed list of computed metrics for this target with 4-decimal precision, e.g. "Accuracy Score: 0.8900" or "R-squared (R2) Score: 0.9421">>
       ===================================
       === END OF MODEL PERFORMANCE SCORECARD ===
       ```
     - Construct a display table containing ALL rows of: {json.dumps(visible_features)} + ground-truth targets + prediction columns `Pred_<target_name>` for all targets.
     - Human-Readable Mapping: Load `category_mappings.json` (which maps label strings to their integer indices, e.g. `{{\"column_name\": {{\"label_name\": 0}}}}`) from the processed-datasets directory if it exists. For any column in the display table, translate its numeric values back to the original category labels using the mapping. Crucially, if a column represents a prediction (i.e. starts with `'Pred_'`, like `Pred_<col>`), extract its original target name (e.g., `col_key = col[5:] if col.startswith('Pred_') else col`) and use the mapping for `col_key`. To translate the values safely, invert the map (`inv_map = {{str(v): k for k, v in mappings[col_key].items()}}`) and apply a robust helper function that handles float-to-int conversion (e.g., convert value `val` to `str(int(float(val)))`), NaNs, and missing values safely by returning `'Unknown'`. Update the display table copy with these mapped strings prior to printing.
     - Configure pandas clean floats printout format (`pd.set_option('display.float_format', lambda x: f'{{x:.4f}}')`) to avoid scientific notation, and configure pandas settings (`pd.set_option('display.max_rows', None)`, `pd.set_option('display.max_columns', None)`, and `pd.set_option('display.width', 1000)`) to print all records cleanly on a single line per row without wrapping columns. Print directly to stdout.

[README.MD RULES]
Generate user manual matching this layout:

# 🤖 Machine Learning Model Pipeline Manual
`Version: 1.0.0` | `Classification: Internal Engineering Document`

---

## 📊 1. Pipeline Architecture & Metadata
| Metadata Parameter | Configured Value Layout |
| :--- | :--- |
| **Modeling Task Strategy** | {problem_type} |
| **Selected Framework Engine** | {chosen_algorithm} |
| **Target Variables ($y$)** | `{json.dumps(chosen_target)}` |
| **Dynamic Image & Container Tag** | `{workspace_folder_id}` |

---

## 🧬 2. Feature Architecture Input Schema
{b3}json
{json.dumps(input_features_only, indent=2)}
{b3}

---

## 🐳 3. Container Execution
### Compile Image
{b3}bash
docker build -t {workspace_folder_id}:latest .
{b3}

### Run container
{b3}bash
docker run --rm --name {workspace_folder_id}-container {workspace_folder_id}:latest
{b3}

---

## 🛡️ 4. Operational Health
- **Model Collapse Check:** Verify predictions change dynamically down rows.
- **Domain Validation:** Confirm target arrays conform to realistic boundaries.

Populate MlArchitectOutput contract perfectly. No markdown wrapping code fence decorators around properties."""

    log.info("Emitting token-optimized script generation request to LLM architect pipeline.")
    try:
        response = structured_llm.invoke(prompt)
        parsed_script_payload: MlArchitectOutput = response["parsed"]
        node_token_count = extract_token_usage(response["raw"])
    except Exception as ai_fault:
        log.error("Platform Fault: LLM Architect failed emitting structured code strings: %s", str(ai_fault))
        return {
            "script_execution_success": False,
            "runtime_stderr": f"LLM Generation Blockage Exception: {str(ai_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "ml_script_architect": 0}
        }

    return {
        "train_script_code": parsed_script_payload.training_script_code,
        "evaluation_script_code": parsed_script_payload.training_script_code,
        "workspace_readme_text": parsed_script_payload.workspace_readme_text,
        "token_count": global_token_count + node_token_count,
        "node_tokens": {**historical_node_tokens, "ml_script_architect": node_token_count}
    }