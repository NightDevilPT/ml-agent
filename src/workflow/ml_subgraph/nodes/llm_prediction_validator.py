"""AI Prediction Quality and Matrix Output Validation Node."""

from typing import Dict, Any
from pydantic import BaseModel, Field

from utils.llm import extract_token_usage, get_llm
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("llm_prediction_validator")


class PredictionValidationContract(BaseModel):
    is_output_sane: bool = Field(
        description="True if model predictions are realistic, dynamic, and show variance down rows. False if model collapse, identical stagnant values, or severe domain errors are present."
    )
    validation_critique: str = Field(
        description="A concise summary critique analyzing why the data rows look correct or pointing out specific prediction anomalies."
    )


def llm_prediction_validator_run(state: MLState) -> Dict[str, Any]:
    """Evaluates the captured stdout terminal data matrix to verify model prediction health."""
    log.section("Invoking LLM Prediction Quality Validator Gate")

    runtime_stdout = state.get("runtime_stdout", "").strip()
    chosen_target = state.get("chosen_target")
    chosen_algorithm = state.get("chosen_algorithm")
    
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})

    if not runtime_stdout:
        log.error("Validator Aborted: No terminal stdout data captured from sandbox container execution.")
        return {
            "model_prediction_accurate": False,
            "runtime_stderr": "[VALIDATION FAILURE]: The holdout evaluation script returned an empty stdout console stream."
        }

    # 🌟 TOKEN OPTIMIZATION: Defaults to gemini-2.0-flash-lite for lightweight token consumption
    llm = get_llm(provider="gemini", temperature=0.0)
    structured_validator = llm.with_structured_output(PredictionValidationContract, include_raw=True)

    # 🌟 COMPRESSED MULTI-TARGET SYSTEM PROMPT DICTION
    prompt = f"""You are an expert Data Scientist reviewing an automated model pipeline validation matrix output.
Determine if the generated model is predicting realistically or if it has experienced model collapse.

[MODEL METADATA]
- Framework Engine: {chosen_algorithm}
- Target Variables (Outputs): {chosen_target}

[CAPTURED WORKSPACE TERMINAL STDOUT INFERENCE MATRIX]
{runtime_stdout}

[STRICT SEMANTIC EVALUATION GUIDELINES]
1. Multi-Target Format: Targets are formatted as array or multiple column vectors matching: {chosen_target}.
2. Check for Model Collapse: Scan predictions vertically. They must be unique and change dynamically down rows. If they are identical or frozen numbers across different inputs, flag it as collapse.
3. Domain Check: Ensure values conform to realistic boundaries (e.g., no negative asset prices for Bitcoin/Stocks).
4. Variance: If rows are unique, dynamic, and realistic, set is_output_sane to True. Otherwise, set to False.

Populate the PredictionValidationContract perfectly."""

    log.info("Submitting captured inference matrix output to AI evaluation gate.")
    try:
        response = structured_validator.invoke(prompt)
        contract: PredictionValidationContract = response["parsed"]
        node_spent = extract_token_usage(response["raw"])
        log.info("Semantic analysis complete. Token count: %d", node_spent)
        
        # If the output data looks corrupted, formulate a feedback error layer for the self-healer node
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
            
        return {
            "model_prediction_accurate": contract.is_output_sane,
            "runtime_stderr": reremed_log if not contract.is_output_sane else None,
            "token_count": global_token_count + node_spent,
            "node_tokens": {**historical_node_tokens, "llm_prediction_validator": node_spent}
        }

    except Exception as ai_fault:
        log.error("Validator Error: Failed getting structured audit from LLM layer: %s", str(ai_fault))
        return {
            "model_prediction_accurate": False,
            "runtime_stderr": f"Semantic Validation Interface Exception: {str(ai_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "llm_prediction_validator": 0}
        }