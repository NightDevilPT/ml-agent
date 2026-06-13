"""
Data Preprocessor Node — Phase 2B: Universal LLM-Guided Preprocessing with Train/Test Splitting.
Dynamically handles high-cardinality features, textual NLP parameters, time-series continuity,
and isolates targets to prevent index shift leakages across arbitrary datasets.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage
from utils.grab_data import get_memory_safe_sample

log = get_logger("data_preprocessor")

class ColumnInstruction(BaseModel):
    action: str = Field(
        description="The processing action: 'parse_datetime', 'one_hot_encode', 'ordinal_encode', 'binary_encode', 'impute_only', 'drop_column', or 'keep_numeric'."
    )
    ordinal_mapping: Dict[str, int] = Field(
        default_factory=dict,
        description="Explicit mapping dictionary for ordinal variables. Otherwise leave empty."
    )

class TransformationBlueprint(BaseModel):
    instructions: Dict[str, ColumnInstruction] = Field(
        description="A dictionary mapping every feature column name to its required numerical conversion rule."
    )


def data_preprocessor_run(state: Dict[str, Any]) -> Dict[str, Any]:
    node_name = "data_preprocessor"
    log.start("Universal Data Preprocessor Node — Phase 2B: Initiating Auto-Ingestion safeguards")
    
    all_files = state.get("all_files", [])
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    target_variable = state.get("target_variable", None)
    
    if not all_files:
        log.error("No valid dataset files found in state tracking context.")
        return {
            "execution_success": False,
            "error_message": "Missing file trajectories. Run dataset_validator first."
        }
        
    clone_workspace = state.get("clone_workspace", "")
    processed_dir = Path(clone_workspace) / "processed_datasets"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    new_processed_paths: List[str] = []
    node_token_count = 0
    
    try:
        for file_str in all_files:
            file_path = Path(file_str)
            log.info("Analyzing column footprints for file: %s", file_path.name)
            
            columns_list, data_sample_string = get_memory_safe_sample(str(file_path), sample_size=10)
            
            # Smart auto-target locator engine
            if not target_variable:
                smart_targets = [c for c in columns_list if c.lower() in ["close", "adj close", "price", "target", "label", "status"]]
                if smart_targets:
                    target_variable = smart_targets[0]
                    log.info("Pipeline auto-detected primary objective target column: [%s]", target_variable)
                else:
                    target_variable = columns_list[-1]
                    log.info("Pipeline fallback auto-detected target column: [%s]", target_variable)
            
            # Read local file dataset matrix blocks
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_str)
            else:
                df = pd.read_csv(file_str)

            # Heuristic Safety Check: Drop absolute unique identifiers (IDs, Hashes, serials) to prevent OOM errors
            dropped_for_high_cardinality = []
            for col in df.columns:
                if col != target_variable and df[col].dtype == 'object':
                    unique_ratio = df[col].nunique() / len(df)
                    if df[col].nunique() > 100 and unique_ratio > 0.5:
                        df.drop(columns=[col], inplace=True)
                        dropped_for_high_cardinality.append(col)
            
            if dropped_for_high_cardinality:
                log.warn("Dropped high-cardinality unique text features to prevent container crashes: %s", dropped_for_high_cardinality)
                columns_list = list(df.columns)

            # Invoke LLM to generate the processing blueprint pattern
            prompt = f"""Role: Expert Data Engineer.
Task: Analyze the following 10-row data snippet to determine exactly how to transform EVERY column into continuous or binary numbers for a machine learning model.

Columns List: {columns_list}
Target Variable Field: {target_variable}
Data Snippet:
{data_sample_string}

Instructions for assigning column actions:
- If a column contains dates/timestamps, mark it as 'parse_datetime'.
- If a column is text with a clear rank/order, mark it as 'ordinal_encode' and provide a numeric map.
- If a column is text with only 2 unique choices, mark it as 'binary_encode'.
- If a column is text with more than 2 unordered categories, mark it as 'one_hot_encode'.
- If a column is already numeric but has missing values, mark it as 'impute_only'.
- If a column is already clean numbers, mark it as 'keep_numeric'."""

            llm = get_llm(temperature=0.0)
            structured_llm = llm.with_structured_output(TransformationBlueprint, include_raw=True)
            
            response = structured_llm.invoke(prompt)
            bp_data: TransformationBlueprint = response["parsed"]
            node_token_count += extract_token_usage(response["raw"])
           
            # Isolate Target column before applying operations
            if target_variable in df.columns:
                target_series = df[target_variable].copy()
                df.drop(columns=[target_variable], inplace=True)
            else:
                raise KeyError(f"Specified objective target field [{target_variable}] missing from data headers.")

            # Apply pattern conversion instructions
            has_time_dimension = False
            for col, rule in bp_data.instructions.items():
                if col not in df.columns:
                    continue
                    
                if df[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna("Unknown")

                if rule.action == "parse_datetime":
                    log.info("Processing Datetime Matrix Split on field: [%s]", col)
                    converted_dt = pd.to_datetime(df[col], errors='coerce')
                    df[f"{col}_year"] = converted_dt.dt.year.fillna(0).astype(int)
                    df[f"{col}_month"] = converted_dt.dt.month.fillna(0).astype(int)
                    df[f"{col}_day"] = converted_dt.dt.day.fillna(0).astype(int)
                    df.drop(columns=[col], inplace=True)
                    has_time_dimension = True
                    
                elif rule.action == "ordinal_encode" and rule.ordinal_mapping:
                    log.info("Mapping Ordinal Structural Categories on field: [%s]", col)
                    df[col] = df[col].map(rule.ordinal_mapping).fillna(0).astype(int)
                    
                elif rule.action == "binary_encode" or rule.action == "one_hot_encode":
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 2:
                        log.info("Applying Binary Layout Cast on field: [%s]", col)
                        df[col] = df[col].map({val: idx for idx, val in enumerate(unique_vals)}).fillna(0).astype(int)
                    else:
                        log.info("Expanding Categorical One-Hot Flags on field: [%s]", col)
                        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

            # Re-attach objective target to absolute far right boundary
            df[target_variable] = target_series
            
            # Coerce remaining boolean objects to clean integers
            for col in df.columns:
                if df[col].dtype == 'bool' or pd.api.types.is_bool_dtype(df[col]):
                    df[col] = df[col].astype(int)
                    
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                raise ValueError(f"Feature transformation processing leaked non-numeric shapes: {non_numeric_cols}")
                
            # Splitting Selector
            if has_time_dimension:
                log.info("Time continuity signature detected. Splitting dataset sequentially to protect temporal validation bounds.")
                split_index = int(len(df) * 0.70)
                df_train = df.iloc[:split_index]
                df_test = df.iloc[split_index:]
            else:
                log.info("Independent row profile detected. Applying standard randomized shuffle allocation.")
                df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                split_index = int(len(df_shuffled) * 0.70)
                df_train = df_shuffled.iloc[:split_index]
                df_test = df_shuffled.iloc[split_index:]
            
            train_out_path = processed_dir / "processed_dataset.csv"
            test_out_path = processed_dir / "test_dataset.csv"
            
            df_train.to_csv(train_out_path, index=False)
            df_test.to_csv(test_out_path, index=False)
            
            new_processed_paths.extend([str(train_out_path.absolute()), str(test_out_path.absolute())])
            log.info("Saved Partition -> Train Rows: %d | Test Rows: %d", len(df_train), len(df_test))
            
    except Exception as process_err:
        log.error("Dynamic column preprocessor validation tracking exception: %s", str(process_err))
        return {
            "execution_success": False,
            "error_message": f"Data preprocessor logic fault: {str(process_err)}"
        }

    updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
    return {
        "processed_files": new_processed_paths,
        "target_variable": target_variable,
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True,
        "error_message": None
    }