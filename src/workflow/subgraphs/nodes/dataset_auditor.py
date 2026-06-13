"""Data Quality Auditor and Verification Gateway Node.

Phase: Closed-Loop Post-Processing Quality Assessment Boundary
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from pydantic import BaseModel, Field

from utils.llm import extract_token_usage, get_llm
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("dataset_auditor")


class AuditAssessmentReport(BaseModel):
    audit_passed: bool = Field(
        description="Set to true if columns are cleanly parsed. Note: Integer types (0, 1) and standard encoded category numbers are completely valid."
    )
    critical_flaws_found: List[str] = Field(
        default_factory=list,
        description="List of factual execution flaws remaining (e.g. raw text strings remaining untouched).",
    )
    remediation_feedback: str = Field(
        description="Clear, actionable instructions sent back to the data engineer node if data validation fails."
    )


def dataset_auditor_run(state: MLState) -> Dict[str, Any]:
    """Runs a strict deterministic verification combined with a structured semantic evaluation loop check."""
    log.section("Dual-Gate Dataset Quality Audit Gateway Initiated")

    all_files = state.get("all_files", [])
    train_path_str = state.get("train_path", "")
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    retry_counters = state.get("retry_counters", {"ingestion_loop": 0})

    if not all_files or not train_path_str:
        log.error("Audit Aborted: Missing required file path tracking pointers.")
        return {
            "is_data_valid": False,
            "consolidation_feedback": "Auditor Configuration Fault: Target registers are vacant.",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "dataset_auditor": 0},
        }

    raw_file_path = Path(all_files[0])
    train_file_path = Path(train_path_str)
    critical_flaws = []

    # GATE 1: Deterministic Code Validation Sweep (0 Tokens)
    try:
        if raw_file_path.suffix.lower() in [".xlsx", ".xls"]:
            df_raw = pd.read_excel(raw_file_path)
        else:
            df_raw = pd.read_csv(raw_file_path)

        df_train = pd.read_csv(train_file_path)
        
        for col in df_train.columns:
            dtype_str = str(df_train[col].dtype)
            # Only flag actual raw unhandled object text fields as structural errors
            if dtype_str in ['object', 'string'] and not pd.api.types.is_numeric_dtype(df_train[col]):
                critical_flaws.append(f"Code Error: Column '{col}' contains unencoded object text string entries.")
            
            null_count = df_train[col].isna().sum()
            if null_count > 0:
                critical_flaws.append(f"Code Error: Column '{col}' contains {null_count} unhandled null/NaN elements.")

        raw_rows_text = df_raw.head(3).to_csv(index=False, sep="|")
        train_rows_text = df_train.head(3).to_csv(index=False, sep="|")

    except Exception as read_fault:
        log.error("Auditor IO Error: Failed reading dataset assets: %s", str(read_fault))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"Auditor IO Extraction Fault: {str(read_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "dataset_auditor": 0},
        }

    # GATE 2: Structured Semantic Alignment Check
    llm = get_llm(provider="gemini", temperature=0.0)
    structured_auditor = llm.with_structured_output(AuditAssessmentReport, include_raw=True)

    prompt = f"""
    You are an expert ML Quality Assurance Auditor. Review these matching data summaries to confirm our preprocessing operations.
    
    VALIDATION GUIDELINES:
    1. The processed dataset MUST contain only numerical tracking variables.
    2. Category features that have been factorized into numbers (e.g. 0.0, 1.0, 2.0) are completely CORRECT. Do NOT flag encoded integers or floats as ambiguous.
    3. Target features (like 'Status') mapped to 0 and 1 are perfectly valid and correct.
    
    [RAW SOURCE RECORDS SAMPLE]
    {raw_rows_text}

    [PROCESSED TRAINING RECORDS SAMPLE]
    {train_rows_text}

    Verify that no raw text entities or string dates are left over in the processed dataset. If the dataset contains only numeric matrices, set audit_passed to True.
    """

    try:
        response = structured_auditor.invoke(prompt)
        audit_report: AuditAssessmentReport = response["parsed"]
        node_spent = extract_token_usage(response["raw"])
        log.info("Dual-Gate verification complete. Tokens consumed: %d.", node_spent)
    except Exception as ai_fault:
        log.error("Platform Fault: LLM auditor failed to evaluate data layers: %s", str(ai_fault))
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"LLM Auditor Exception: {str(ai_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "dataset_auditor": 0},
        }

    all_discovered_flaws = list(set(critical_flaws + audit_report.critical_flaws_found))

    # Determine final pipeline trajectory outcome states
    if audit_report.audit_passed and not all_discovered_flaws:
        log.info("✅ DUAL-GATE AUDIT PASSED: Training matrix matches downstream training configurations.")
        return {
            "is_data_valid": True,
            "consolidation_feedback": None,
            "token_count": global_token_count + node_spent,
            "node_tokens": {**historical_node_tokens, "dataset_auditor": node_spent},
        }
    else:
        # Increment our validation loop tracking register safely to track ceilings
        current_loops = retry_counters.get("ingestion_loop", 0)
        updated_counters = {**retry_counters, "ingestion_loop": current_loops + 1}
        
        log.warn("❌ DUAL-GATE AUDIT FAILED: Optimization flaws detected. Iteration loop index: %d", current_loops + 1)
        
        return {
            "is_data_valid": False,
            "consolidation_feedback": f"Auditor Corrections: {audit_report.remediation_feedback}",
            "retry_counters": updated_counters,
            "token_count": global_token_count + node_spent,
            "node_tokens": {**historical_node_tokens, "dataset_auditor": node_spent},
        }