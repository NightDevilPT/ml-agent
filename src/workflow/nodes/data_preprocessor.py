"""
Data Preprocessor Node — Phase 2B: Universal LLM-Guided Preprocessing with Train/Test Splitting.
Dynamically handles high-cardinality features, textual NLP parameters, time-series continuity,
and isolates targets to prevent index shift leakages across arbitrary datasets.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage
from utils.grab_data import get_memory_safe_sample

log = get_logger("data_preprocessor")

# ================================================================
# Constants
# ================================================================
MIN_ROWS_REQUIRED = 10
MIN_COLUMNS_REQUIRED = 2
MAX_ONEHOT_CATEGORIES = 8
MAX_FREQUENCY_ENCODING_THRESHOLD = 15
HIGH_CARDINALITY_DROP_RATIO = 0.7
HIGH_CARDINALITY_DROP_MIN_UNIQUE = 200


# ================================================================
# Pydantic Schemas
# ================================================================
class ColumnInstruction(BaseModel):
    action: str = Field(
        description="Processing action: 'parse_datetime', 'one_hot_encode', 'ordinal_encode', "
                    "'binary_encode', 'frequency_encode', 'clean_numeric', 'impute_only', "
                    "'drop_column', or 'keep_numeric'."
    )
    ordinal_mapping: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping for ordinal variables OR cleaning instructions for clean_numeric "
                    "(e.g., {'strip_units': 'kms', 'remove_commas': true, 'strip_currency': true})."
    )


class TransformationBlueprint(BaseModel):
    instructions: Dict[str, ColumnInstruction] = Field(
        description="Dictionary mapping every feature column name to its required numerical conversion rule."
    )


# ================================================================
# Helper Functions
# ================================================================
def _validate_input_dataframe(df: pd.DataFrame, file_name: str) -> None:
    """Comprehensive input validation with clear error messages."""
    
    if len(df) == 0:
        raise ValueError(f"Empty dataset: '{file_name}' contains zero rows.")
    
    if len(df) < MIN_ROWS_REQUIRED:
        log.warn("Dataset '%s' has only %d rows (minimum: %d). Proceeding with caution.",
                 file_name, len(df), MIN_ROWS_REQUIRED)
    
    if len(df.columns) < MIN_COLUMNS_REQUIRED:
        raise ValueError(
            f"Dataset has only {len(df.columns)} columns. "
            f"Need at least {MIN_COLUMNS_REQUIRED} (features + target)."
        )
    
    # Detect completely empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        log.warn("Completely empty columns detected and dropped: %s", empty_cols)
        df.drop(columns=empty_cols, inplace=True)
    
    # Detect duplicate column names
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicated_cols:
        raise ValueError(f"Duplicate column names found: {duplicated_cols}")


def _profile_dataset(df: pd.DataFrame, target_variable: Optional[str]) -> Dict[str, Any]:
    """Generate comprehensive dataset profile for logging and debugging."""
    
    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": {},
        "outliers": {}
    }
    
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            profile["missing_values"][col] = {
                "count": int(missing),
                "percentage": round(missing / len(df) * 100, 2)
            }
        profile["column_types"][col] = str(df[col].dtype)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == target_variable:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            profile["outliers"][col] = {
                "count": int(outlier_count),
                "percentage": round(outlier_count / len(df) * 100, 2)
            }
    
    return profile


def _detect_target_variable(df: pd.DataFrame, columns_list: List[str]) -> str:
    """Smart auto-target detection with multiple fallback strategies."""
    
    smart_targets = [
        c for c in columns_list 
        if c.lower().strip() in [
            "close", "adj close", "price", "target", "label", 
            "status", "class", "output", "result", "outcome",
            "sales", "revenue", "churn", "response", "y"
        ]
    ]
    
    if smart_targets:
        target = smart_targets[0]
        log.info("Target auto-detected by name: [%s]", target)
        return target
    
    candidate_cols = [c for c in columns_list if df[c].nunique() > 1]
    
    if candidate_cols:
        uniqueness = {c: df[c].nunique() for c in candidate_cols}
        target = min(uniqueness, key=uniqueness.get)
        log.info("Target auto-detected by uniqueness: [%s] (%d unique values)", 
                 target, uniqueness[target])
        return target
    
    target = columns_list[-1]
    log.info("Target fallback to last column: [%s]", target)
    return target


def _validate_llm_blueprint(
    bp_data: TransformationBlueprint, 
    df_columns: List[str]
) -> TransformationBlueprint:
    """Ensure LLM blueprint covers ALL columns with fallback logic."""
    
    missing_cols = [col for col in df_columns if col not in bp_data.instructions]
    
    if missing_cols:
        log.warn("LLM blueprint missed %d columns. Applying fallback auto-detection.", 
                 len(missing_cols))
        
        instructions_dict = dict(bp_data.instructions)
        
        for col in missing_cols:
            if pd.api.types.is_datetime64_any_dtype(bp_data):
                instructions_dict[col] = ColumnInstruction(action="parse_datetime")
                log.info("  Fallback [%s] → parse_datetime", col)
            elif pd.api.types.is_numeric_dtype(bp_data):
                instructions_dict[col] = ColumnInstruction(action="keep_numeric")
                log.info("  Fallback [%s] → keep_numeric", col)
            elif pd.api.types.is_bool_dtype(bp_data):
                instructions_dict[col] = ColumnInstruction(action="binary_encode")
                log.info("  Fallback [%s] → binary_encode", col)
            else:
                instructions_dict[col] = ColumnInstruction(action="frequency_encode")
                log.info("  Fallback [%s] → frequency_encode", col)
        
        bp_data = TransformationBlueprint(instructions=instructions_dict)
    
    return bp_data


def _validate_final_dataframe(df: pd.DataFrame, stage: str) -> None:
    """Post-processing validation to ensure no bad data leaks through."""
    
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"[{stage}] Non-numeric columns remaining: {non_numeric}. "
            f"All columns must be numeric before splitting."
        )
    
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        log.warn("[%s] NaN values found in columns: %s. Filling with 0.", stage, nan_cols)
        df.fillna(0, inplace=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            log.warn("[%s] Infinite values found in column: %s. Replacing with 0.", stage, col)
            df[col] = df[col].replace([np.inf, -np.inf], 0)


def _detect_task_type(target_series: pd.Series) -> str:
    """Auto-detect whether the target suggests classification or regression."""
    
    unique_values = target_series.nunique()
    total_values = len(target_series)
    
    if unique_values <= 15:
        min_freq = target_series.value_counts().min()
        if min_freq >= total_values * 0.01:
            return "classification"
    
    if pd.api.types.is_float_dtype(target_series) and unique_values > 20:
        return "regression"
    
    if pd.api.types.is_integer_dtype(target_series) and unique_values <= 10:
        return "classification"
    
    return "regression"


# ================================================================
# Main Node Entrypoint
# ================================================================
def data_preprocessor_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main preprocessor execution:
    1. Validate input files
    2. Profile dataset
    3. Detect target variable
    4. Get LLM transformation blueprint
    5. Execute transformations
    6. Validate output
    7. Split 70/30 and save
    """
    node_name = "data_preprocessor"
    log.start("Data Preprocessor — Phase 2B: Universal Preprocessing Engine")
    
    all_files = state.get("all_files", [])
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    target_variable = state.get("target_variable", None)
    
    if not all_files:
        log.error("No valid dataset files found in state context.")
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
            log.section(f"Processing: {file_path.name}")
            
            # ──────────────────────────────────────────────────
            # STEP 1: Load and validate raw data
            # ──────────────────────────────────────────────────
            log.info("Loading and validating raw dataset...")
            
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_str)
            else:
                df = pd.read_csv(file_str)
            
            _validate_input_dataframe(df, file_path.name)
            
            columns_list = list(df.columns)
            log.info("Raw dataset: %d rows × %d columns", len(df), len(columns_list))
            
            # ──────────────────────────────────────────────────
            # STEP 2: Dataset profiling
            # ──────────────────────────────────────────────────
            dataset_profile = _profile_dataset(df, target_variable)
            
            if dataset_profile["duplicate_rows"] > 0:
                log.warn("Found %d duplicate rows.", dataset_profile["duplicate_rows"])
            
            if dataset_profile["missing_values"]:
                log.info("Missing values detected in %d columns.", 
                         len(dataset_profile["missing_values"]))
            
            if dataset_profile["outliers"]:
                log.info("Outliers detected in %d numeric columns.", 
                         len(dataset_profile["outliers"]))
            
            # ──────────────────────────────────────────────────
            # STEP 3: Target variable detection
            # ──────────────────────────────────────────────────
            if not target_variable:
                target_variable = _detect_target_variable(df, columns_list)
            else:
                if target_variable not in df.columns:
                    log.error("Specified target [%s] not found in columns: %s", 
                              target_variable, columns_list)
                    return {
                        "execution_success": False,
                        "error_message": f"Target '{target_variable}' not in dataset."
                    }
            
            if df[target_variable].nunique() == len(df):
                log.warn("Target [%s] has all unique values — possible ID column.", 
                         target_variable)
            
            task_type = _detect_task_type(df[target_variable])
            log.info("Detected task type: %s (target: [%s], unique values: %d)", 
                     task_type, target_variable, df[target_variable].nunique())
            
            # ──────────────────────────────────────────────────
            # STEP 4: Get memory-safe sample for LLM
            # ──────────────────────────────────────────────────
            log.info("Extracting memory-safe sample for LLM analysis...")
            columns_list, data_sample_string = get_memory_safe_sample(
                str(file_path), sample_size=10
            )
            
            # ──────────────────────────────────────────────────
            # STEP 5: Drop high-cardinality ID columns
            # ──────────────────────────────────────────────────
            dropped_cols = []
            for col in df.columns:
                if col == target_variable:
                    continue
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    unique_ratio = unique_count / len(df)
                    if unique_count > HIGH_CARDINALITY_DROP_MIN_UNIQUE and \
                       unique_ratio > HIGH_CARDINALITY_DROP_RATIO:
                        dropped_cols.append({
                            "column": col,
                            "unique_values": unique_count,
                            "reason": "High cardinality identifier"
                        })
            
            for item in dropped_cols:
                df.drop(columns=[item["column"]], inplace=True)
                log.warn("Dropped: [%s] (%d unique values, %.1f%% unique ratio)", 
                         item["column"], item["unique_values"], 
                         item["unique_values"] / len(df) * 100)
            
            if dropped_cols:
                columns_list = list(df.columns)
            
            # ──────────────────────────────────────────────────
            # STEP 6: Get LLM transformation blueprint
            # ──────────────────────────────────────────────────
            log.section("Invoking LLM for transformation blueprint")
            
            prompt = f"""Role: Expert Data Engineer.
Task: Analyze the following 10-row data snippet to determine exactly how to transform EVERY column into continuous or binary numbers for a machine learning model.

Columns List: {columns_list}
Target Variable Field: {target_variable}
Data Snippet:
{data_sample_string}

CRITICAL RULES FOR ASSIGNING COLUMN ACTIONS:

1. DATES/TIMESTAMPS:
   - If a column contains dates or timestamps → 'parse_datetime'

2. MIXED NUMBER+TEXT (like "45,000 kms", "$80,000", "2000kg", "₹2,85,000"):
   - Use 'clean_numeric' and specify cleaning instructions in ordinal_mapping.
   - Available cleaning flags: "strip_currency" (true/false), "strip_units" (text to remove like "kms", "kg", "cc"), "remove_commas" (true/false).
   - Example: {{"strip_currency": true, "strip_units": "kms", "remove_commas": true}}

3. TEXT WITH RANK/ORDER (Low<Medium<High, Beginner<Intermediate<Expert):
   - Use 'ordinal_encode' with explicit mapping: {{"Low": 0, "Medium": 1, "High": 2}}

4. TEXT WITH EXACTLY 2 UNIQUE VALUES:
   - Use 'binary_encode'

5. TEXT WITH 3-{MAX_ONEHOT_CATEGORIES} UNORDERED CATEGORIES:
   - Use 'one_hot_encode'

6. TEXT WITH MORE THAN {MAX_ONEHOT_CATEGORIES} CATEGORIES:
   - MUST use 'frequency_encode' (NEVER one_hot_encode for high cardinality!)

7. NUMERIC WITH MISSING VALUES:
   - Use 'impute_only'

8. CLEAN NUMERIC (no issues):
   - Use 'keep_numeric'

9. UNIQUE IDENTIFIERS (all values different, like names, IDs, hashes):
   - Use 'drop_column'

Return instructions for EVERY column in the Columns List above."""

            llm = get_llm(temperature=0.0)
            structured_llm = llm.with_structured_output(
                TransformationBlueprint, include_raw=True
            )
            
            response = structured_llm.invoke(prompt)
            bp_data: TransformationBlueprint = response["parsed"]
            node_token_count += extract_token_usage(response["raw"])
            
            # Validate blueprint completeness
            bp_data = _validate_llm_blueprint(bp_data, list(df.columns))
            
            log.info("LLM blueprint generated with %d column instructions.", 
                     len(bp_data.instructions))
            
            # ──────────────────────────────────────────────────
            # STEP 7: Isolate target column
            # ──────────────────────────────────────────────────
            if target_variable not in df.columns:
                raise KeyError(f"Target [{target_variable}] missing from dataframe.")
            
            target_series = df[target_variable].copy()
            df.drop(columns=[target_variable], inplace=True)
            
            # ──────────────────────────────────────────────────
            # STEP 8: Execute transformations
            # ──────────────────────────────────────────────────
            log.section("Executing column transformations")
            has_time_dimension = False
            
            for col, rule in bp_data.instructions.items():
                if col not in df.columns:
                    continue
                
                # Handle missing values first
                if df[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                        log.info("  [%s] → imputed NaN with median", col)
                    else:
                        df[col] = df[col].fillna("Unknown")
                        log.info("  [%s] → imputed NaN with 'Unknown'", col)
                
                # --- ACTION: parse_datetime ---
                if rule.action == "parse_datetime":
                    log.info("  [%s] → datetime split (year/month/day/dayofweek)", col)
                    converted_dt = pd.to_datetime(df[col], errors='coerce')
                    df[f"{col}_year"] = converted_dt.dt.year.fillna(0).astype(int)
                    df[f"{col}_month"] = converted_dt.dt.month.fillna(0).astype(int)
                    df[f"{col}_day"] = converted_dt.dt.day.fillna(0).astype(int)
                    df[f"{col}_dayofweek"] = converted_dt.dt.dayofweek.fillna(0).astype(int)
                    df.drop(columns=[col], inplace=True)
                    has_time_dimension = True
                
                # --- ACTION: clean_numeric (NEW) ---
                elif rule.action == "clean_numeric":
                    log.info("  [%s] → cleaning numeric string", col)
                    
                    strip_info = rule.ordinal_mapping if rule.ordinal_mapping else {}
                    temp_series = df[col].astype(str)
                    
                    if strip_info.get("strip_currency"):
                        temp_series = temp_series.str.replace(
                            r'[$€£¥₹]', '', regex=True
                        )
                    
                    if strip_info.get("strip_units"):
                        unit = str(strip_info["strip_units"])
                        temp_series = temp_series.str.replace(unit, '', regex=False)
                    
                    if strip_info.get("remove_commas"):
                        temp_series = temp_series.str.replace(',', '')
                    
                    df[col] = pd.to_numeric(temp_series, errors='coerce').fillna(0)
                
                # --- ACTION: ordinal_encode ---
                elif rule.action == "ordinal_encode" and rule.ordinal_mapping:
                    log.info("  [%s] → ordinal encoding (%d categories)", 
                             col, len(rule.ordinal_mapping))
                    df[col] = df[col].map(rule.ordinal_mapping).fillna(0).astype(int)
                
                # --- ACTION: binary_encode ---
                elif rule.action == "binary_encode":
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) == 2:
                        log.info("  [%s] → binary encoding", col)
                        mapping = {val: idx for idx, val in enumerate(unique_vals)}
                        df[col] = df[col].map(mapping).fillna(0).astype(int)
                    else:
                        log.warn("  [%s] → expected binary (2 values), got %d. Factorizing.", 
                                 col, len(unique_vals))
                        df[col] = pd.factorize(df[col])[0]
                
                # --- ACTION: one_hot_encode ---
                elif rule.action == "one_hot_encode":
                    unique_count = df[col].nunique()
                    
                    if unique_count > MAX_FREQUENCY_ENCODING_THRESHOLD:
                        log.info("  [%s] → frequency encoding (%d unique values > %d threshold)", 
                                 col, unique_count, MAX_FREQUENCY_ENCODING_THRESHOLD)
                        freq_map = df[col].value_counts(normalize=True).to_dict()
                        df[col] = df[col].map(freq_map).fillna(0).astype(float)
                    
                    elif unique_count > MAX_ONEHOT_CATEGORIES:
                        log.info("  [%s] → one-hot encoding with top %d categories (%d total)", 
                                 col, MAX_ONEHOT_CATEGORIES, unique_count)
                        top_cats = df[col].value_counts().nlargest(MAX_ONEHOT_CATEGORIES).index.tolist()
                        df[col] = df[col].apply(lambda x: x if x in top_cats else 'OTHER')
                        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
                    
                    elif unique_count == 2:
                        log.info("  [%s] → binary encoding (exactly 2 categories)", col)
                        df[col] = pd.factorize(df[col])[0]
                    
                    else:
                        log.info("  [%s] → one-hot encoding (%d categories)", col, unique_count)
                        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
                
                # --- ACTION: frequency_encode ---
                elif rule.action == "frequency_encode":
                    log.info("  [%s] → frequency encoding", col)
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0).astype(float)
                
                # --- ACTION: impute_only ---
                elif rule.action == "impute_only":
                    log.info("  [%s] → impute only (keeping as numeric)", col)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # --- ACTION: keep_numeric ---
                elif rule.action == "keep_numeric":
                    log.info("  [%s] → keep as-is (numeric)", col)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # --- ACTION: drop_column ---
                elif rule.action == "drop_column":
                    log.info("  [%s] → dropped", col)
                    df.drop(columns=[col], inplace=True)
            
            # ──────────────────────────────────────────────────
            # STEP 9: Re-attach target and coerce types
            # ──────────────────────────────────────────────────
            df[target_variable] = target_series
            
            # Coerce booleans to integers
            bool_cols = df.select_dtypes(include=['bool']).columns
            for col in bool_cols:
                df[col] = df[col].astype(int)
            
            # Coerce any remaining object columns to numeric
            obj_cols = df.select_dtypes(include=['object']).columns
            for col in obj_cols:
                if col != target_variable:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # ──────────────────────────────────────────────────
            # STEP 10: Final validation
            # ──────────────────────────────────────────────────
            _validate_final_dataframe(
                df.drop(columns=[target_variable]), 
                "Pre-Split"
            )
            
            log.info("Transformation complete: %d rows × %d columns", 
                     len(df), len(df.columns))
            
            # ──────────────────────────────────────────────────
            # STEP 11: Train/Test Split
            # ──────────────────────────────────────────────────
            log.section("Creating 70/30 Train/Test Split")
            
            if has_time_dimension:
                log.info("Time-series detected → sequential split")
                split_index = int(len(df) * 0.70)
                df_train = df.iloc[:split_index].reset_index(drop=True)
                df_test = df.iloc[split_index:].reset_index(drop=True)
            
            elif task_type == "classification":
                log.info("Classification detected → stratified split")
                from sklearn.model_selection import train_test_split
                
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.30, random_state=42, stratify=y
                )
                
                df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
                df_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
            
            else:
                log.info("Standard regression → random shuffle split")
                df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                split_index = int(len(df_shuffled) * 0.70)
                df_train = df_shuffled.iloc[:split_index].reset_index(drop=True)
                df_test = df_shuffled.iloc[split_index:].reset_index(drop=True)
            
            # Validate split
            if len(df_train) < 2:
                raise ValueError(f"Training set too small: {len(df_train)} rows.")
            if len(df_test) < 1:
                raise ValueError(f"Test set too small: {len(df_test)} rows.")
            if list(df_train.columns) != list(df_test.columns):
                raise ValueError("Column mismatch between train and test sets!")
            
            # ──────────────────────────────────────────────────
            # STEP 12: Save processed datasets
            # ──────────────────────────────────────────────────
            train_out_path = processed_dir / "processed_dataset.csv"
            test_out_path = processed_dir / "test_dataset.csv"
            
            df_train.to_csv(train_out_path, index=False)
            df_test.to_csv(test_out_path, index=False)
            
            new_processed_paths.extend([
                str(train_out_path.absolute()), 
                str(test_out_path.absolute())
            ])
            
            log.info("Train set: %d rows → %s", len(df_train), train_out_path.name)
            log.info("Test set:  %d rows → %s", len(df_test), test_out_path.name)
            
    except Exception as process_err:
        log.error("Preprocessor fault: %s", str(process_err))
        import traceback
        log.error(traceback.format_exc())
        return {
            "execution_success": False,
            "error_message": f"Data preprocessor logic fault: {str(process_err)}"
        }
    
    updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
    
    log.end("Preprocessing pipeline completed successfully!")
    
    return {
        "processed_files": new_processed_paths,
        "target_variable": target_variable,
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True,
        "error_message": None
    }