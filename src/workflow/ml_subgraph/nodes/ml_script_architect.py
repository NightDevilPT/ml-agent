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
    problem_type = state.get("problem_type") or "Regression"
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

[CRITICAL CODE SYSTEM INSTRUCTIONS]
1. Core Layout: Include `import argparse`, `import pandas as pd`, `import joblib`, and `from pathlib import Path` at top.
2. Paths: Compute relative paths exactly via:
   BASE_DIR = Path(__file__).resolve().parent
   train_path = BASE_DIR / "{train_folder_name}" / "train_dataset.csv"
   test_path = BASE_DIR / "{train_folder_name}" / "test_dataset.csv"
   model_path = BASE_DIR / "model.joblib"
3. Execution Logic:
   - `--mode train`: Load relative `train_path`, drop targets {json.dumps(chosen_target)}, fit model '{chosen_algorithm}', save to relative `model_path`.
   - `--mode evaluate`: Load relative `model_path`, load first 10 rows from relative `test_path`. Compute predictions array.
4. 🌟 NO SCIENTIFIC NOTATION RULE: In `--mode evaluate`, you MUST explicitly configure pandas to display floating-point format conversions cleanly without exponential notation via: `pd.set_option('display.float_format', lambda x: f'{{x:.4f}}')`. 
   - Construct a display table containing these columns: {json.dumps(visible_features)} + ground-truth targets + custom prediction target column references. Print directly to stdout via `print(display_df.to_string(index=False))`.

[STRICT README.MD TEMPLATE STRUCTURING RULES]
For `workspace_readme_text`, generate an enterprise-grade document matching this EXACT structural layout:

# 🤖 Machine Learning Model Pipeline Manual
`Version: 1.0.0` | `Classification: Internal Engineering Document`

Comprehensive operational runbook detailing data architecture, baseline algorithm configurations, and containerized runtime infrastructure execution for the automated ML-Agent pipeline.

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
The model isolates target features and accepts the following transformed numeric variables to calculate vector inferences:
{b3}json
{json.dumps(input_features_only, indent=2)}
{b3}

---

## 🐳 3. Simplified Container Execution Runbook
### Step 1: Compile the Isolated Environment Image
{b3}bash
docker build -t {workspace_folder_id}:latest .
{b3}

### Step 2: Initialize Model Training and Evaluation Execution Sequence
{b3}bash
docker run --rm --name {workspace_folder_id}-container {workspace_folder_id}:latest
{b3}

---

## 📺 4. Expected Output Matrix Format
{b3}text
Prints a fully combined data grid containing inputs, ground truths, and model predictions cleanly side-by-side.
{b3}

---

## 🛡️ 5. Operational Health & Error Bounding
- **Model Collapse Check:** Verify predictions change dynamically down rows. Identical outputs signify localized gradient failures.
- **Domain Validation:** Confirm target arrays conform to realistic boundaries (no negative asset prices for financial domains).

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