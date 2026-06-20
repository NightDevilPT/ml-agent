"""Single-File Auto-Adaptive Validation & Multi-Encoding Engine.

Phase: LLM Schema Recipe Generation & Native Python Execution Matrix
"""

import json
import re
import math
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
def execute_column_recipe_engine(df: pd.DataFrame, recipe: ColumnRecipe, is_target: bool = False) -> Optional[pd.Series]:
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

        numeric_series = pd.to_numeric(series, errors="coerce")
        return numeric_series.astype("float64")

    # Strategy D: Clean string categories/text variables safely before encoding sweeps (retaining missing values as NaN)
    return series.astype(str).str.strip().apply(lambda x: np.nan if str(x).lower() in ["nan", "null", "none", ""] else x)


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

    df_peek = df_raw.head(5)

    # Call Structured LLM to establish granular processing configuration rules
    llm = get_llm(provider="gemini", temperature=0.0)
    structured_recipe_agent = llm.with_structured_output(DatasetValidationRecipe, include_raw=True)

    prompt = f"""
    You are a Principal Lead Data Infrastructure Platform Engineer.    Analyze this raw dataset preview snapshot sample (CSV format):
    {df_peek.to_csv(index=False)}

    Your task is to evaluate the feature columns and output an explicit parsing and encoding recipe JSON.
    
    Encoding Instructions:
    1. Identify the primary prediction target column.
    2. If a column contains unique tracking tokens, hashes, or identification codes (like Patient_ID, User_ID, Row_ID, name, description), set text_encoding_strategy to 'drop'.
    3. Category Cardinality & Encoding Rules:
       - If a categorical/text column has high cardinality (many unique string values, e.g. more than 15 unique values, such as 'name', 'model', 'city', 'hospital', 'company'), set text_encoding_strategy to 'ordinal' (label encoding). This prevents expanding the feature space with hundreds of sparse dummy columns and makes it much easier for tree models to learn.
       - Set text_encoding_strategy to 'one_hot' ONLY for categorical columns with low cardinality (15 or fewer unique values, such as 'Gender', 'Fuel_Type', 'State', 'Status').
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

        # Step 1: Clean and normalize columns (retaining NaNs)
        for col_recipe in recipe_blueprint.column_recipes:
            raw_col_name = col_recipe.column_name.strip()
            if raw_col_name in df_raw.columns:
                is_target = (raw_col_name == recipe_blueprint.recommended_target)
                processed_series = execute_column_recipe_engine(df_raw, col_recipe, is_target)
                
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

        # Step 2: Handle target cleaning and dropping invalid rows
        target_col = recipe_blueprint.recommended_target
        if target_col in df_base.columns:
            before_drop = len(df_base)
            target_recipe = next((r for r in recipe_blueprint.column_recipes if r.column_name == target_col), None)
            
            if target_recipe and target_recipe.logical_type == "numeric":
                df_base[target_col] = pd.to_numeric(df_base[target_col], errors="coerce")
                df_base = df_base.dropna(subset=[target_col])
                # Drop rows where target is <= 0 for numeric targets that are positive-value (e.g. price, close, counts)
                if df_base[target_col].median() > 0:
                    df_base = df_base[df_base[target_col] > 0]
            else:
                df_base = df_base.dropna(subset=[target_col])
                
            log.info("Target cleaning: dropped %d rows with missing or invalid target values.", before_drop - len(df_base))

        # Step 3: Remove extremely noisy rows with too many missing values (threshold: at least 65% columns must be valid)
        num_cols = len(df_base.columns)
        if num_cols > 1:
            min_valid_cols = math.ceil(0.65 * num_cols)
            before_drop_thresh = len(df_base)
            df_base = df_base.dropna(thresh=min_valid_cols)
            log.info("Missingness cleanup: dropped %d rows with too many missing values (threshold: %d/%d valid columns).", 
                     before_drop_thresh - len(df_base), min_valid_cols, num_cols)

        # Step 4: Perform missing value imputation on remaining rows
        for col_recipe in recipe_blueprint.column_recipes:
            col_name = col_recipe.column_name.strip()
            if col_name in df_base.columns:
                series = df_base[col_name]
                if series.isna().sum() > 0:
                    strategy = col_recipe.missing_value_strategy.lower()
                    if col_recipe.logical_type == "numeric":
                        if strategy == "mean":
                            df_base[col_name] = series.fillna(series.mean())
                        elif strategy == "mode":
                            mode_val = series.mode()
                            df_base[col_name] = series.fillna(mode_val[0] if not mode_val.empty else 0.0)
                        elif strategy == "constant":
                            fallback_val = float(col_recipe.unparsable_text_fallback) if col_recipe.unparsable_text_fallback.replace('.', '', 1).isdigit() else 0.0
                            df_base[col_name] = series.fillna(fallback_val)
                        else:  # median
                            df_base[col_name] = series.fillna(series.median() if not series.isna().all() else 0.0)
                    else:  # categorical/datetime/boolean
                        # For categorical, map median/mean/mode to mode of column to avoid generating 'Unknown' classes unnecessarily
                        if strategy in ["mode", "median", "mean"]:
                            mode_val = series.mode()
                            df_base[col_name] = series.fillna(mode_val[0] if not mode_val.empty else "Unknown")
                        else:
                            df_base[col_name] = series.fillna("Unknown")

        # Step 5: Native Mathematical Encoding Operations
        df_final = df_base.copy()

        # Execute Ordinal Category Mapping
        category_mappings = {}
        for col in ordinal_targets:
            log.info("Applying programmatic Ordinal Encoding on column: '%s'", col)
            labels, uniques = pd.factorize(df_final[col])
            df_final[col] = labels.astype("float64")
            category_mappings[col] = {str(x): int(i) for i, x in enumerate(uniques)}

        # Execute One-Hot Category Expansion Matrix
        if one_hot_targets:
            log.info("Applying programmatic One-Hot Dummy Encoding on columns: %s", one_hot_targets)
            df_final = pd.get_dummies(df_final, columns=one_hot_targets, drop_first=True, dtype="float64")

        # Handle Categorical Target encoding if not numeric
        target_col = recipe_blueprint.recommended_target
        if target_col in df_final.columns:
            target_recipe = next((r for r in recipe_blueprint.column_recipes if r.column_name == target_col), None)
            if target_recipe and target_recipe.logical_type != "numeric" and not pd.api.types.is_numeric_dtype(df_final[target_col]):
                log.info("Encoding classification label metrics on target feature: '%s'", target_col)
                labels, uniques = pd.factorize(df_final[target_col])
                df_final[target_col] = labels.astype("float64")
                category_mappings[target_col] = {str(x): int(i) for i, x in enumerate(uniques)}

        # Save category mappings JSON
        mappings_path = train_output_path.parent / "category_mappings.json"
        try:
            import json
            with open(mappings_path, "w", encoding="utf-8") as f:
                json.dump(category_mappings, f, indent=4, ensure_ascii=False)
            log.info("Category mappings saved successfully to: %s", mappings_path)
        except Exception as e:
            log.error("Failed saving category mappings: %s", str(e))

        # Step 6: Advanced Datetime Feature Engineering Layer
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