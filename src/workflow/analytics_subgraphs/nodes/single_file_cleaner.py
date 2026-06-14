"""Single-File Auto-Adaptive Validation & Multi-Encoding Engine.

Phase: LLM Schema Recipe Generation & Native Python Execution Matrix
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from utils.llm import extract_token_usage, get_llm
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("single_file_cleaner")


# ==============================================================================
# Pydantic Schemas: The Granular Column Transformation Recipe Contract
# ==============================================================================
class ColumnRecipe(BaseModel):
    column_name: str = Field(
        description="The exact name of the column in the dataset (case-sensitive)."
    )
    logical_type: Literal["numeric", "text", "datetime", "boolean", "category"] = Field(
        description="Target ML data group for this feature."
    )
    characters_to_strip: List[str] = Field(
        default_factory=list,
        description="Specific characters mixed into numbers to clean out (e.g. ['$', ',', 'kms', '%']).",
    )
    unparsable_text_fallback: str = Field(
        default="0",
        description="Fallback numeric string value if text noise is found (e.g., '0' for 'Ask For Price').",
    )
    missing_value_strategy: Literal["median", "mean", "mode", "constant"] = Field(
        default="median",
        description="Strategy for handling empty, missing, or NaN values.",
    )
    text_encoding_strategy: Literal["one_hot", "ordinal", "drop", "none"] = Field(
        default="none",
        description="How text categories should be turned into numbers. Use 'one_hot' for unordered words, 'ordinal' for ranked categories like stages/grades, and 'drop' for unique IDs.",
    )


class DatasetValidationRecipe(BaseModel):
    dataset_summary: str = Field(
        description="Brief high-level summary of the dataset layout."
    )
    recommended_target: str = Field(
        description="The primary predictive target variable column name."
    )
    is_ready_for_training: bool = Field(
        description="Set to true if this dataset can proceed after cleaning."
    )
    column_recipes: List[ColumnRecipe] = Field(
        description="The transformation recipe generated for every single column."
    )


# ==============================================================================
# The Deterministic Python Execution Engine
# ==============================================================================
def execute_column_recipe_engine(df: pd.DataFrame, recipe: ColumnRecipe) -> Optional[pd.Series]:
    """Applies strict, explicit data transformations on a raw column based on its recipe blueprint."""
    col = recipe.column_name
    series = df[col].copy()

    if recipe.text_encoding_strategy == "drop":
        return None

    # Strategy A: Convert strings into explicit datetime objects
    if recipe.logical_type == "datetime":
        return pd.to_datetime(series, errors="coerce")

    # Strategy B: Factor text metrics down to explicit booleans
    if recipe.logical_type == "boolean":
        return (
            series.astype(str)
            .str.lower()
            .str.strip()
            .isin(["true", "1", "1.0", "yes", "y", "t"])
        )

    # Strategy C: Extract pure numerical fields out from dirty string data
    if recipe.logical_type == "numeric":
        # 🌟 ENHANCEMENT: Aggressively eliminate inline punctuation, spacing, and commas up front
        # This completely resolves the Indian comma numbering format variations (e.g., 4,25,000 -> 425000)
        series = series.astype(str).str.replace(",", "", regex=False).str.strip()

        for symbol in sorted(recipe.characters_to_strip, key=len, reverse=True):
            if symbol != ",":  # Handled natively above, but skip gracefully if present in list
                series = series.str.replace(symbol, "", regex=False)

        series = series.str.strip()

        if recipe.unparsable_text_fallback is not None:
            series = series.apply(
                lambda x: (
                    recipe.unparsable_text_fallback
                    if not re.search(r"\d", str(x))
                    else x
                )
            )

        numeric_series = pd.to_numeric(series, errors="coerce")

        if numeric_series.isna().sum() > 0:
            strategy = recipe.missing_value_strategy.lower()
            if strategy == "mean":
                numeric_series = numeric_series.fillna(numeric_series.mean())
            elif strategy == "mode":
                mode_val = numeric_series.mode()
                numeric_series = numeric_series.fillna(
                    mode_val[0] if not mode_val.empty else 0.0
                )
            elif strategy == "constant":
                fallback_val = float(recipe.unparsable_text_fallback) if recipe.unparsable_text_fallback.replace('.', '', 1).isdigit() else 0.0
                numeric_series = numeric_series.fillna(fallback_val)
            else:
                numeric_series = numeric_series.fillna(
                    numeric_series.median() if not numeric_series.isna().all() else 0.0
                )

        return numeric_series.astype("float64")

    # Strategy D: Clean string categories/text variables safely before encoding sweeps
    return series.astype(str).str.strip().fillna("Unknown")


# ==============================================================================
# Node Entry Control Point
# ==============================================================================
def single_file_cleaner_run(state: MLState) -> Dict[str, Any]:
    """Runs a 7-step profiling process, cleans metrics, handles advanced datetime engineering,

    converts all text categories into numerical structures natively, and writes a model-ready training file.
    """
    log.section("Structured Validation Recipe Engine Initiated")

    all_files = state.get("all_files", [])
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})

    if not all_files:
        log.error("Execution Aborted: Shared tracking registers contain no valid source files.")
        return {
            "is_data_valid": False,
            "consolidation_feedback": "Error: Missing data entries inside state vector.",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": 0},
        }

    raw_file_path = Path(all_files[0])
    workspace_path = Path(state.get("clone_workspace", ""))
    train_output_path = workspace_path / "processed-datasets" / "train_dataset.csv"

    try:
        if raw_file_path.suffix.lower() in [".xlsx", ".xls"]:
            df_raw = pd.read_excel(raw_file_path)
        else:
            df_raw = pd.read_csv(raw_file_path)
        log.info("Base Ingestion Layer Completed. Raw Shape Profile: %s", str(df_raw.shape))
    except Exception as io_err:
        log.error("Ingestion Error: Failed to open targeted spreadsheet data: %s", str(io_err))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"IO Extraction Fault: {str(io_err)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": 0},
        }

    df_peek = df_raw.head(10)

    # Call Structured LLM to establish granular processing configuration rules
    llm = get_llm(provider="gemini", temperature=0.0)
    structured_recipe_agent = llm.with_structured_output(DatasetValidationRecipe, include_raw=True)

    prompt = f"""
    You are a Principal Lead Data Infrastructure Platform Engineer. Analyze this raw dataset preview snapshot sample:
    {df_peek.to_json(orient='records', indent=2)}

    Your task is to evaluate the feature columns and output an explicit parsing and encoding recipe JSON.
    
    Encoding Instructions:
    1. Identify the primary prediction target column.
    2. If a column contains unique tracking tokens or identification codes (like Patient_ID, User_ID, Row_ID), set text_encoding_strategy to 'drop'.
    3. If a column contains plain text categories with no logical order (like Gender, State, Cancer_Type, Fuel_Type), set text_encoding_strategy to 'one_hot'.
    4. If a column contains text with a clear ranking or progression sequence (like Stage I, Stage II, Stage III or Low, Medium, High), set text_encoding_strategy to 'ordinal'.
    5. For numeric metrics or datetimes, set text_encoding_strategy to 'none'.
    
    Populate the DatasetValidationRecipe configuration contract perfectly.
    """

    log.info("Emitting dataset evaluation request to the structured LLM recipe profiler.")
    try:
        response = structured_recipe_agent.invoke(prompt)
        recipe_blueprint: DatasetValidationRecipe = response["parsed"]
        node_spent = extract_token_usage(response["raw"])
        log.info("Structural profiling configuration contract compiled. Tokens spent: %d", node_spent)
    except Exception as ai_fault:
        log.error("Platform Fault: AI failed emitting a structured validation contract: %s", str(ai_fault))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"LLM Generation Crash: {str(ai_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": 0},
        }

    # 3. Native Data Processing & Mathematical Encoding Execution Matrix
    try:
        df_base = pd.DataFrame()
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        
        datetime_cols = []
        one_hot_targets = []
        ordinal_targets = []

        # Step 1: Clean, normalize, and drop specified columns
        for col_recipe in recipe_blueprint.column_recipes:
            raw_col_name = col_recipe.column_name.strip()
            if raw_col_name in df_raw.columns:
                processed_series = execute_column_recipe_engine(df_raw, col_recipe)
                
                if processed_series is None:
                    log.info("--> Dropping column space completely: '%s'", raw_col_name)
                    continue
                    
                df_base[raw_col_name] = processed_series
                
                # Sort out remaining columns into encoding target buckets
                if col_recipe.logical_type == "datetime":
                    datetime_cols.append(raw_col_name)
                elif col_recipe.text_encoding_strategy == "one_hot" and raw_col_name != recipe_blueprint.recommended_target:
                    one_hot_targets.append(raw_col_name)
                elif col_recipe.text_encoding_strategy == "ordinal" and raw_col_name != recipe_blueprint.recommended_target:
                    ordinal_targets.append(raw_col_name)

        # Step 2: Native Mathematical Encoding Operations
        df_final = df_base.copy()

        # Execute Ordinal Category Mapping
        for col in ordinal_targets:
            log.info("Applying programmatic Ordinal Encoding on column: '%s'", col)
            df_final[col] = pd.factorize(df_final[col])[0].astype("float64")

        # Execute One-Hot Category Expansion Matrix
        if one_hot_targets:
            log.info("Applying programmatic One-Hot Dummy Encoding on columns: %s", one_hot_targets)
            df_final = pd.get_dummies(df_final, columns=one_hot_targets, drop_first=True, dtype="float64")

        # Step 3: Handle Target Formatting explicitly
        target_col = recipe_blueprint.recommended_target
        if target_col in df_final.columns and not pd.api.types.is_numeric_dtype(df_final[target_col]):
            log.info("Encoding classification label metrics on target feature: '%s'", target_col)
            df_final[target_col] = pd.factorize(df_final[target_col])[0]

        # Step 4: Advanced Datetime Feature Engineering Layer
        for col in datetime_cols:
            if col in df_final.columns:
                log.info("Feature Engineering: Decomposing datetime column '%s' into numeric elements.", col)
                parsed_dates = pd.to_datetime(df_final[col], errors="coerce")
                
                df_final[f"{col}_year"] = parsed_dates.dt.year.fillna(parsed_dates.dt.year.median() if not parsed_dates.dt.year.isna().all() else 2026).astype("float64")
                df_final[f"{col}_month"] = parsed_dates.dt.month.fillna(parsed_dates.dt.month.median() if not parsed_dates.dt.month.isna().all() else 1).astype("float64")
                df_final[f"{col}_day"] = parsed_dates.dt.day.fillna(parsed_dates.dt.day.median() if not parsed_dates.dt.day.isna().all() else 1).astype("float64")
                
                df_final.drop(columns=[col], inplace=True)
                log.info("--> Dropped raw unreadable column '%s' and generated numerical sub-features.", col)

        # Remove row duplicates
        df_final.drop_duplicates(inplace=True)

        # 4. Save clean, 100% numerical training dataset file
        train_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(train_output_path, index=False)

        log.info("Final processed training schema types:\n%s", df_final.dtypes.to_string())
        log.info("Perfect mathematical training dataset successfully written to: %s", train_output_path)
        log.end("Single-file automated preprocessing pipeline complete.")

        return {
            "is_data_valid": recipe_blueprint.is_ready_for_training,
            "consolidation_feedback": None,
            "train_path": str(train_output_path.absolute()),
            "target_recommendations": [{"target_name": recipe_blueprint.recommended_target, "description": "Primary verified target column."}],
            "token_count": global_token_count + node_spent,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": node_spent},
        }

    except Exception as runtime_engine_fault:
        log.error("Pipeline Engine Crash: Processing loops failed: %s", str(runtime_engine_fault))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"Execution Processing Exception: {str(runtime_engine_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "single_file_cleaner": 0},
        }