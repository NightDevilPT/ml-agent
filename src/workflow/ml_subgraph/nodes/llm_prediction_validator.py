"""AI Prediction Quality and Matrix Output Validation Node."""

from typing import Dict, Any
from pydantic import BaseModel, Field

from utils.llm import extract_token_usage, get_llm
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("llm_prediction_validator")


class TargetValidationDetail(BaseModel):
    explanation: str = Field(
        description="Detailed explanation/critique for this target's predictions and performance metrics."
    )
    rating: float = Field(
        description="A target-specific prediction quality score rating between 0.0 (unusable) and 1.0 (highly accurate/good)."
    )
    metrics: Dict[str, float] = Field(
        description="Dictionary of all accuracy and error metrics shown in the scorecard for this target (e.g. Accuracy Score, Precision Score, Mean Squared Error, R-squared (R2) Score, etc. exactly as shown)."
    )


class PredictionValidationContract(BaseModel):
    is_output_sane: bool = Field(
        description="True if all model predictions are realistic and sane. False if severe flaws, model collapse, or unacceptable metrics exist."
    )
    validation_critique: str = Field(
        description="Concise critique summarizing validation issues or confirming complete correctness across all targets."
    )
    target_evaluations: Dict[str, TargetValidationDetail] = Field(
        description="A dictionary mapping each target column name to its detailed evaluation metrics and explanation."
    )


import re

def llm_prediction_validator_run(state: MLState) -> Dict[str, Any]:
    """Evaluates the captured stdout scorecard block to verify model performance health."""
    log.section("Invoking LLM Prediction Quality Validator Gate")

    runtime_stdout = state.get("runtime_stdout", "").strip()
    chosen_target = state.get("chosen_target")
    chosen_algorithm = state.get("chosen_algorithm")
    problem_type = state.get("problem_type") or "Regression"
    
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})

    if not runtime_stdout:
        log.error("Validator Aborted: No terminal stdout data captured from sandbox container execution.")
        return {
            "model_prediction_accurate": False,
            "runtime_stderr": "[VALIDATION FAILURE]: The holdout evaluation script returned an empty stdout console stream.",
            "model_performance_rating": {},
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "llm_prediction_validator": 0}
        }

    # Filter out timestamp patterns (e.g., "[2026-06-20 18:27:05]" or "2026-06-20 18:27:05") and clean up the lines
    timestamp_pattern = re.compile(r'\[?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]?')
    clean_lines = []
    for line in runtime_stdout.splitlines():
        # Remove any timestamp matches
        cleaned = timestamp_pattern.sub('', line).strip()
        # Remove leftover trailing or leading punctuation like hyphens or colons
        cleaned = re.sub(r'^[:\-\s]+|[:\-\s]+$', '', cleaned).strip()
        if not cleaned:
            continue
        clean_lines.append(cleaned)

    extracted_parts = []
    
    # 1. Grab training status line if present
    for line in clean_lines:
        if "model trained and saved" in line.lower():
            extracted_parts.append(line)
            break
            
    # 2. Extract the MODEL PERFORMANCE SCORECARD block
    scorecard_lines = []
    in_scorecard = False
    scorecard_keywords = [
        "scorecard", "strategy", "framework", "target", "accuracy", "precision", 
        "recall", "f1", "error", "squared", "r-squared", "r2", "mae", "mse", "===="
    ]
    scan_count = 0
    for line in clean_lines:
        if "=== START OF MODEL PERFORMANCE SCORECARD ===" in line or "=== MODEL PERFORMANCE SCORECARD ===" in line:
            in_scorecard = True
        if in_scorecard:
            scan_count += 1
            if scan_count > 100:  # Prevent scanning more than 100 lines past the scorecard header
                in_scorecard = False
                break
            # Only append lines containing valid scorecard keywords
            if any(kw in line.lower() for kw in scorecard_keywords):
                # Omit the wrapper start/end tags themselves from entering validation log prompt
                if "start of model" not in line.lower() and "end of model" not in line.lower():
                    scorecard_lines.append(line)
            if "=== END OF MODEL PERFORMANCE SCORECARD ===" in line or ("===================================" in line and len(scorecard_lines) > 1):
                in_scorecard = False
                break
                
    if scorecard_lines:
        extracted_parts.extend(scorecard_lines)
    else:
        # Fallback to last 15 lines if no scorecard block explicitly demarcated
        extracted_parts.extend(clean_lines[-15:])

    validation_log_snippet = "\n".join(extracted_parts)

    llm = get_llm(provider="gemini", temperature=0.0)
    structured_validator = llm.with_structured_output(PredictionValidationContract, include_raw=True)

    prompt = f"""Review the validation scorecard to ensure the model trained successfully.

[METADATA]
- Strategy: {problem_type}
- Framework: {chosen_algorithm}
- Target: {chosen_target}

[VALIDATION SCORECARD]
{validation_log_snippet}

[SEMANTIC GUIDELINES]
1. Multi-Target Evaluation Alignment:
   - Identify the task type (Classification or Regression) for EACH target listed in the scorecard.
   - For EACH target, evaluate it according to its specific task type:
     * Classification targets: evaluate classification metrics (Accuracy, Precision, Recall, F1 Score).
     * Regression targets: evaluate regression metrics (R-squared, MSE, MAE).
2. Sane Metrics and Sanity Check:
   - For Classification targets:
     - For binary classification, confirm Accuracy >= 0.55.
     - For multi-class classification (> 2 classes), confirm Accuracy is substantially better than random guessing (e.g. Accuracy >= 0.45).
     - If any classification target fails these thresholds, set is_output_sane to False.
   - For Regression targets:
     - Confirm R-squared (R2) is positive (R2 > 0).
     - If any regression target has a non-positive R2, set is_output_sane to False.
3. Target-Specific Goodness Evaluation:
   - For EACH target, populate the details in the `target_evaluations` dict.
   - Provide a rating score between 0.0 (unusable/terrible) and 1.0 (perfectly accurate/highly optimized) under target's rating.
   - Extract and populate all metric names and numeric values shown in scorecard under target's metrics (e.g. "Accuracy Score": 0.5185, "R-squared (R2) Score": 0.9107).
   - Write a detailed target-specific explanation under target's explanation explaining its rating and metrics.

Populate the PredictionValidationContract perfectly."""

    log.info("Submitting compressed inference scorecard to AI evaluation gate.")
    try:
        response = structured_validator.invoke(prompt)
        contract: PredictionValidationContract = response["parsed"]
        node_spent = extract_token_usage(response["raw"])
        log.info("Semantic analysis complete. Token count: %d", node_spent)
        
        reremed_log = None
        if not contract.is_output_sane:
            log.error("Validation Alarm Raised: Model output is broken. Critique: %s", contract.validation_critique)
            r_loops = state.get("retry_counters", {}).get("generation_loop", 0)
            reremed_log = (
                f"[CRITICAL SEMANTIC FAILURE - REFACTOR ATTEMPT #{r_loops + 1}]:\n"
                f"The code ran without crashing, but the model predictions are flawed!\n"
                f"AI Auditor Critique: {contract.validation_critique}\n"
                f"Please inspect feature extraction, target mappings, scaling transformations, or model fitting steps."
            )
            
        target_evals_dict = {}
        for target_name, detail in contract.target_evaluations.items():
            target_evals_dict[target_name] = {
                "explanation": detail.explanation,
                "rating": detail.rating,
                "metrics": detail.metrics
            }
            
        return {
            "model_prediction_accurate": contract.is_output_sane,
            "runtime_stderr": reremed_log if not contract.is_output_sane else None,
            "model_performance_rating": target_evals_dict,
            "token_count": global_token_count + node_spent,
            "node_tokens": {**historical_node_tokens, "llm_prediction_validator": node_spent}
        }

    except Exception as ai_fault:
        log.error("Validator Error: Failed getting structured audit from LLM layer: %s", str(ai_fault))
        return {
            "model_prediction_accurate": False,
            "runtime_stderr": f"Semantic Validation Interface Exception: {str(ai_fault)}",
            "model_performance_rating": {},
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "llm_prediction_validator": 0}
        }