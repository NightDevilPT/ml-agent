"""Single-File Auto-Adaptive Validation & Multi-Encoding Engine.

Phase: LLM Schema Recipe Generation & Native Python Execution Matrix

Improvements over v1:
  - Sends full dataset profile to LLM (dtypes, null%, nunique, describe, skewness)
    instead of just df.head(5), so encoding/imputation decisions are evidence-based.
  - Prompt covers 9 explicit preprocessing phases:
      1. Inspection
      2. Column dropping (nulls >60%, constants, duplicates, IDs)
      3. Target variable handling
      4. Text / mixed-type cleaning
      5. Datetime feature engineering
      6. Missing value imputation (median vs mean vs mode)
      7. Outlier capping (IQR method, features only)
      8. Categorical encoding (OHE vs label, mappings saved)
      9. Final validation assertions + shape report
  - Generated script must pass hard assertions before saving output.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from utils.llm import extract_token_usage, get_llm
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("single_file_cleaner")


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class PreprocessingScriptBlueprint(BaseModel):
    recommended_target: str = Field(
        description=(
            "The primary target variable column name identified for prediction. "
            "Must be a column present in the raw dataset."
        )
    )
    problem_type: str = Field(
        description=(
            "Either 'classification' or 'regression', inferred from the target column's "
            "distribution (few unique values / object dtype → classification; continuous "
            "numeric with many unique values → regression)."
        )
    )
    dataset_summary: str = Field(
        description=(
            "A brief description of the dataset: what it represents, the target variable, "
            "key feature columns, and any notable data quality issues observed."
        )
    )
    columns_to_drop: list[str] = Field(
        description=(
            "List of column names the script will drop before modelling. Include: "
            "ID/key columns, >60% null columns, zero-variance constants, and exact duplicates."
        )
    )
    python_code: str = Field(
        description=(
            "The complete, execution-ready Python preprocessing script. "
            "Must implement all 9 phases described in the prompt and end with "
            "hard assertion checks before saving outputs."
        )
    )


# ---------------------------------------------------------------------------
# Dataset profiler — builds a rich, LLM-readable summary
# ---------------------------------------------------------------------------

def _build_dataset_profile(df: pd.DataFrame) -> str:
    """Return an optimized, highly compact markdown table profile of the raw DataFrame."""

    lines: list[str] = []
    lines.append(f"SHAPE: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")
    lines.append("SAMPLE ROWS (first 2):")
    lines.append(df.head(2).to_csv(index=False))
    lines.append("")
    lines.append("HEADERS AND TYPES:")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col}: {dtype}")
    lines.append("")
    lines.append("COLUMN PROFILE:")
    lines.append("| Column | Dtype | Null% | NUnique | Skew | Stats/Top Values |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for col in df.columns:
        dtype = df[col].dtype
        null_pct = round((df[col].isnull().mean() * 100), 2)
        nunique = df[col].nunique()
        
        # Determine skewness for numeric columns
        skew_str = "N/A"
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                skew_val = df[col].skew()
                if not pd.isna(skew_val):
                    skew_str = str(round(skew_val, 2))
            except Exception:
                pass
        
        # Build Stats / Top Values description
        stats_str = ""
        flags = []
        if nunique == len(df) and nunique > 1:
            flags.append("POTENTIAL ID")
        if nunique <= 1:
            flags.append("CONSTANT")
        if null_pct > 60:
            flags.append("HIGH NULL")

        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                desc = df[col].describe()
                mean_val = round(desc['mean'], 2) if not pd.isna(desc['mean']) else "NaN"
                std_val = round(desc['std'], 2) if not pd.isna(desc['std']) else "NaN"
                min_val = round(desc['min'], 2) if not pd.isna(desc['min']) else "NaN"
                max_val = round(desc['max'], 2) if not pd.isna(desc['max']) else "NaN"
                stats_str = f"mean={mean_val}, std={std_val}, min={min_val}, max={max_val}"
            except Exception:
                stats_str = "Error calculating stats"
        else:
            try:
                top_vals = df[col].value_counts().head(3).to_dict()
                stats_str = str(top_vals)
            except Exception:
                stats_str = "Error calculating top values"

        if flags:
            stats_str = f"[{', '.join(flags)}] " + stats_str

        # Add row to markdown table
        lines.append(f"| {col} | {dtype} | {null_pct}% | {nunique} | {skew_str} | {stats_str} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(profile: str, relative_raw_path: str) -> str:
    return f"""Write a Python script to preprocess the raw dataset.
=== DATASET PROFILE ===
{profile}
=== END PROFILE ===

LOAD PATH: '{relative_raw_path}'
OUTPUT DIR: 'processed-datasets/'
ALLOWED IMPORTS: os, re, json, pandas, numpy, sklearn only.
Decide based on the DATASET PROFILE, not assumptions.

━━━ PIPELINE STEPS ━━━
1. LOAD & AUDIT: Load CSV/Excel. Print shape, dtypes, nulls, nunique. Store original shape.
2. DROP COLUMNS: Drop if (a) ID column (nunique==len(df) AND name has 'id','index','serial','key','no.'), (b) Nulls > 60%, (c) nunique <= 1, (d) exact duplicate of other column. NEVER drop target or descriptive text (e.g. names/categories, keep for Step 8).
3. TARGET: Assign target_col = '<name>'. Drop rows where target is NaN. If target is classification (string/object), apply LabelEncoder and store in `category_mappings` (cast values to native Python `int`). If numeric, cast to float. Print stats.
4. CLEAN MIXED-TYPE/TEXT: 
   - Formatting ('45,000 kms', '$1,200', '15%'): regex strip non-numeric (keep digits/dots) and cast to float. Revert if >50% fail.
   - Long strings (>3 words avg): truncate to first 2 words (e.g. 'Honda Civic 2019' -> 'Honda Civic').
   - Booleans ('Yes'/'No', 'Y'/'N'): map to 1/0.
5. DATETIME: Detect columns parseable as dates (name has 'date','time','year','month' etc., skip 'year' if already numeric). pd.to_datetime, extract year/month/day/dayofweek, drop original, drop NaT rows.
6. IMPUTE: Numeric skew >1.0 -> median, else mean. Categorical/Object/Boolean -> mode (or 'Unknown'). Compute skew AFTER step 4. Never impute target.
7. CAP OUTLIERS: Numeric features only (exclude target_col, binary, or IQR==0). Clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Do not drop rows.
8. ENCODE: features only. Encode all remaining object, category, string, and boolean feature columns (excluding target_col):
   - If col dtype is 'object', 'category', or 'string':
     * If nunique <= 15: Use `pd.get_dummies(df, columns=[col], drop_first=False, dtype=int)` (or equivalent) to one-hot encode, and cast any boolean OHE output columns to integer immediately.
     * If nunique > 15: Use `LabelEncoder` to label encode, and save mapping in `category_mappings[col]` with all keys and values cast to native Python types (string keys, native `int` values).
   - If col dtype is 'bool' or boolean:
     * Cast to `int` using `df[col] = df[col].astype(int)`.
   - Ensure that NO object, category, or boolean columns remain in the output dataframe features.
9. FINAL VALIDATION & SAVE:
   - Run these exact checks. On failure, print error and sys.exit(1):
     1) assert df.isnull().sum().sum() == 0, 'NaNs remain'
     2) assert len(df.select_dtypes(include=['object', 'bool', 'category']).columns) == 0, f'Non-numeric columns remain: {{df.select_dtypes(include=["object","bool","category"]).columns.tolist()}}'
     3) assert len(df) > 0, 'DataFrame is empty'
     4) assert target_col in df.columns, 'Target missing'
     5) assert np.issubdtype(df[target_col].dtype, np.number), 'Target not numeric'
   - Print summary: original vs cleaned shape, target & task type, columns dropped, encoding strategies, success confirmation.
   - Save CSV to 'processed-datasets/train_dataset.csv' and category_mappings JSON to 'processed-datasets/category_mappings.json'.

━━━ RULES ━━━
- category_mappings = {{}} at script top.
- Hardcoded decisions, no user input, try/except on I/O.
- Standard script execution: python data-process.py. Pass `include=['object', 'str']` to `select_dtypes` to avoid warnings.
Output ONLY the runnable Python script. No markdown, explanations, or TODOs.
"""


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------

def single_file_cleaner_run(state: MLState) -> Dict[str, Any]:
    """Generates and runs a dataset-specific Python preprocessing script
    to clean and encode raw columns inside an isolated Docker container."""

    log.section("Structured Validation Preprocessing Script Generation Initiated")

    all_files = state.get("all_files", [])
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})

    # ------------------------------------------------------------------
    # Guard: need at least one file
    # ------------------------------------------------------------------
    if not all_files:
        log.error("Execution Aborted: No source files found in state.")
        return {
            "is_data_valid": False,
            "consolidation_feedback": "Error: Missing data entries inside state vector.",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": 0},
        }

    raw_file_path = Path(all_files[0])
    workspace_path = Path(state.get("clone_workspace", ""))
    train_output_path = workspace_path / "processed-datasets" / "train_dataset.csv"
    mappings_path = workspace_path / "processed-datasets" / "category_mappings.json"

    train_output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load raw file
    # ------------------------------------------------------------------
    try:
        if raw_file_path.suffix.lower() in [".xlsx", ".xls"]:
            df_raw = pd.read_excel(raw_file_path)
        else:
            df_raw = pd.read_csv(raw_file_path)
        log.info("Raw data loaded. Shape: %s", str(df_raw.shape))
    except Exception as io_err:
        log.error("Failed to load raw file: %s", str(io_err))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"IO Extraction Fault: {str(io_err)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": 0},
        }

    # ------------------------------------------------------------------
    # Build rich dataset profile for LLM
    # ------------------------------------------------------------------
    log.info("Building dataset profile for LLM prompt...")
    dataset_profile = _build_dataset_profile(df_raw)

    # Relative path for the generated script to use when loading data
    try:
        relative_raw_path = raw_file_path.relative_to(workspace_path).as_posix()
    except Exception:
        relative_raw_path = f"datasets/{raw_file_path.name}"

    # ------------------------------------------------------------------
    # Call LLM with structured output
    # ------------------------------------------------------------------
    llm = get_llm(provider="gemini", temperature=0.0)
    structured_recipe_agent = llm.with_structured_output(
        PreprocessingScriptBlueprint, include_raw=True
    )

    prompt = _build_prompt(dataset_profile, relative_raw_path)

    log.info("Sending dataset profile to LLM for preprocessing script generation...")
    node_spent = 0
    try:
        response = structured_recipe_agent.invoke(prompt)
        blueprint: PreprocessingScriptBlueprint = response["parsed"]
        node_spent = extract_token_usage(response["raw"])
        log.info(
            "Blueprint received. Target: '%s' | Problem: %s | Tokens: %d",
            blueprint.recommended_target,
            blueprint.problem_type,
            node_spent,
        )
        log.info("Columns LLM will drop: %s", blueprint.columns_to_drop)
    except Exception as ai_fault:
        log.error("LLM blueprint generation failed: %s", str(ai_fault))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"LLM Preprocessing Code Generation Crash: {str(ai_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": 0},
        }

    # ------------------------------------------------------------------
    # Write generated script and Dockerfile
    # ------------------------------------------------------------------
    script_path = workspace_path / "data-process.py"
    dockerfile_path = workspace_path / "Dockerfile.preprocess"

    workspace_folder_id = workspace_path.name.lower()
    image_tag = f"{workspace_folder_id}-preprocess:latest"
    container_name = f"{workspace_folder_id}-preprocess-container"

    try:
        # Write data-process.py
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(blueprint.python_code)
        log.info("Saved data-process.py → %s", script_path)

        # Write Dockerfile
        dockerfile_content = (
            "FROM python:3.12-slim\n"
            "WORKDIR /workspace\n"
            "RUN pip install --no-cache-dir pandas openpyxl scikit-learn numpy\n"
            "COPY . .\n"
            'CMD ["python", "data-process.py"]\n'
        )
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        log.info("Saved Dockerfile.preprocess → %s", dockerfile_path)

        # --------------------------------------------------------------
        # STEP 1: Build Docker image
        # --------------------------------------------------------------
        log.info("Building Docker image [%s]...", image_tag)
        build_proc = subprocess.run(
            ["docker", "build", "-t", image_tag, "-f", "Dockerfile.preprocess", "."],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if build_proc.returncode != 0:
            log.error("Docker image build failed:\n%s", build_proc.stderr)
            return {
                "is_data_valid": False,
                "consolidation_feedback": (
                    f"Preprocessor Docker image build failed:\n{build_proc.stderr}"
                ),
                "token_count": global_token_count + node_spent,
                "node_tokens": {**historical_node_tokens, "single_file_cleaner": node_spent},
            }

        # --------------------------------------------------------------
        # STEP 2: Run preprocessing container
        # --------------------------------------------------------------
        log.info("Running preprocessing container [%s]...", container_name)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

        run_proc = subprocess.run(
            ["docker", "run", "--name", container_name, image_tag],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        # Always capture and log stdout so we can see the script's summary report
        if run_proc.stdout:
            log.info("Container stdout:\n%s", run_proc.stdout)

        if run_proc.returncode != 0:
            error_detail = run_proc.stderr or run_proc.stdout or "No output captured."
            log.error("Container runtime failed:\n%s", error_detail)
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
            return {
                "is_data_valid": False,
                "consolidation_feedback": (
                    f"Preprocessor container runtime failed:\n{error_detail}"
                ),
                "token_count": global_token_count + node_spent,
                "node_tokens": {**historical_node_tokens, "single_file_cleaner": node_spent},
            }

        # --------------------------------------------------------------
        # STEP 3: Copy outputs out of container
        # --------------------------------------------------------------
        log.info("Copying preprocessed outputs from container...")
        shutil_dest = workspace_path / "processed-datasets"
        if shutil_dest.exists():
            shutil.rmtree(shutil_dest, ignore_errors=True)

        cp_proc = subprocess.run(
            ["docker", "cp", f"{container_name}:/workspace/processed-datasets", str(workspace_path)],
            capture_output=True,
            check=False,
        )

        # Cleanup container and image
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
        subprocess.run(["docker", "rmi", image_tag], capture_output=True, check=False)

        if cp_proc.returncode != 0:
            log.error("Failed to copy outputs from container.")
            return {
                "is_data_valid": False,
                "consolidation_feedback": "Failed copying preprocessed datasets from Docker container.",
                "token_count": global_token_count + node_spent,
                "node_tokens": {**historical_node_tokens, "single_file_cleaner": node_spent},
            }

        # --------------------------------------------------------------
        # STEP 4: Verify output files exist and are valid
        # --------------------------------------------------------------
        if not train_output_path.exists():
            raise FileNotFoundError(
                f"train_dataset.csv was not generated at: {train_output_path}"
            )

        # Quick sanity read — confirm file is a valid non-empty CSV
        df_out = pd.read_csv(train_output_path)
        if df_out.empty:
            raise ValueError("Generated train_dataset.csv is empty.")

        # Warn if any non-numeric columns somehow survived
        leftover_obj = df_out.select_dtypes(include=["object", "str"]).columns.tolist()
        if leftover_obj:
            log.warning(
                "Non-numeric columns still present in output (should have been caught "
                "by script assertions): %s",
                leftover_obj,
            )

        log.info(
            "Preprocessing complete. Output shape: %s | Saved to: %s",
            str(df_out.shape),
            train_output_path,
        )

        return {
            "is_data_valid": True,
            "consolidation_feedback": None,
            "train_path": str(train_output_path.absolute()),
            "mappings_path": str(mappings_path.absolute()) if mappings_path.exists() else None,
            "data_process_script_code": blueprint.python_code,
            "target_recommendations": [
                {
                    "target_name": blueprint.recommended_target,
                    "problem_type": blueprint.problem_type,
                    "description": blueprint.dataset_summary,
                    "columns_dropped": blueprint.columns_to_drop,
                }
            ],
            "output_shape": list(df_out.shape),
            "token_count": global_token_count + node_spent,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": node_spent},
        }

    except Exception as run_err:
        log.error("Preprocessing execution failed: %s", str(run_err))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"Execution Processing Exception: {str(run_err)}",
            "token_count": global_token_count + node_spent,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": node_spent},
        }