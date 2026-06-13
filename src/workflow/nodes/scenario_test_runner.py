"""
Scenario Test Runner Node — Phase 7: Behavioral Constraint Verification

Evaluates model inferences against adversarial edge cases using an LLM evaluator.
Leverages structured outputs via Pydantic and enforces strict type alignment to prevent XGBoost object crashes.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import joblib
from pydantic import BaseModel, Field

# Local module imports
from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage

log = get_logger("scenario_test_runner")

# ================================================================
# Pydantic Schema for Structured Evaluation Output
# ================================================================
class BehavioralEvaluationGrading(BaseModel):
    status: str = Field(
        description="The ultimate verification grade verdict. Must be exactly 'PASSED' or 'FAILED'."
    )
    engineering_justification: str = Field(
        description="A short reason sentence detailing why the predictive vector matches or violates the QA rule specification."
    )


# ================================================================
# Main Node Entrypoint
# ================================================================
def scenario_test_runner_run(state: Dict[str, Any]) -> Dict[str, Any]:
    node_name = "scenario_test_runner"
    
    # Halting Guardrail: Stop if upstream processing collapsed
    if state.get("execution_success") is False:
        return {
            "execution_success": False,
            "error_message": state.get("error_message", "Halted test runner: Upstream pipeline nodes are in a failed state.")
        }
        
    log.start("Scenario Test Runner Node — Phase 7: Running Mock Behavioral Validations")
    
    clone_workspace = state.get("clone_workspace", "")
    test_scenarios = state.get("test_scenarios", [])
    selected_config = state.get("selected_algorithm_config", {})
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    
    model_path = Path(clone_workspace) / "model.joblib"
    if not model_path.exists():
        log.error("Trained model artifact binary file missing from workspace root directory.")
        return {
            "execution_success": False,
            "error_message": "Model binary artifact file not found on host disk space."
        }
        
    node_token_count = 0
    try:
        # Load the compiled binary object object safely back into local memory
        model = joblib.load(model_path)
        
        # 🌟 LOOKUP REAL TYPES: Read 1 row of the processed data to steal its true column data types
        processed_dir = Path(clone_workspace) / "processed_datasets"
        processed_files = [f for f in processed_dir.iterdir() if f.is_file() and f.suffix.lower() == '.csv']
        reference_dtypes = {}
        if processed_files:
            df_ref = pd.read_csv(processed_files[0], nrows=1)
            reference_dtypes = df_ref.dtypes.to_dict()

        failed_cases = []
        log.section("Evaluating Adversarial Assertion Test Matrix")
        
        for idx, scenario in enumerate(test_scenarios, start=1):
            name = scenario.get("scenario_name")
            features = scenario.get("input_features_matrix", {})
            assertion = scenario.get("expected_behavioral_assertion")
            
            # Construct single row dataframe matching feature constraints
            df_mock = pd.DataFrame([features])
            
            # Force alignment of prediction dataframe column order to match model expectancies
            if hasattr(model, "feature_names_in_"):
                df_mock = df_mock[list(model.feature_names_in_)]
            
            # 🌟 STSTRUCTURAL DATA TYPE ALIGNMENT FIX: 
            # Convert string numbers (like "16400.5") into real floats/ints based on the training file types
            for col in df_mock.columns:
                if col in reference_dtypes:
                    try:
                        df_mock[col] = df_mock[col].astype(reference_dtypes[col])
                    except Exception:
                        # Fallback: if it's numeric but inferred as object, force to numeric
                        df_mock[col] = pd.to_numeric(df_mock[col], errors='coerce').fillna(0)
            
            # Execute inference
            if hasattr(model, "predict"):
                prediction = model.predict(df_mock)[0]
            else:
                raise AttributeError("Loaded joblib model binary does not contain a standard .predict() invocation signature.")
                
            # Invoke structured evaluation tool schema
            check_prompt = f"""Role: AI Model Auditor.
Task: Evaluate if this model's prediction outcome aligns with engineering QA expectations.

Model Architecture: {selected_config.get('algorithm_name')}
Scenario Track: {name}
Input Record Values: {json.dumps(features)}
Expected Behavior Rule Specification: {assertion}
Actual Numerical Prediction Result: {prediction}

Instructions: Determine if the prediction result makes logical, mathematical sense according to the expected behavior rule description."""

            llm = get_llm(temperature=0.0)
            structured_llm = llm.with_structured_output(BehavioralEvaluationGrading, include_raw=True)
            
            response = structured_llm.invoke(check_prompt)
            evaluation: BehavioralEvaluationGrading = response["parsed"]
            node_token_count += extract_token_usage(response["raw"])
            
            if evaluation.status.upper() == "FAILED":
                log.warn("Test Case FAILED: %s | Reason: %s", name, evaluation.engineering_justification)
                failed_cases.append(
                    f"Case #{idx} [{name}]: expected '{assertion}', got predictive unit response '{prediction}'. "
                    f"Reason: {evaluation.engineering_justification}"
                )
            else:
                log.info("Test Case PASSED: %s", name)

        if failed_cases:
            report_str = "\n".join(failed_cases)
            log.warn("Behavioral testing concluded with localized assertion failures.")
            
            updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
            return {
                "test_failure_report": report_str, 
                "token_count": global_token_count + node_token_count,
                "node_tokens": updated_node_tokens,
                "execution_success": True,
                "error_message": None
            }
            
    except Exception as eval_fault:
        log.error("Internal testing runner loop encountered a syntax or system fault: %s", str(eval_fault))
        return {
            "execution_success": False,
            "error_message": f"Testing execution engine runtime crash: {str(eval_fault)}"
        }

    updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
    log.end("All localized behavioral edge-case tests successfully passed validation rules!")
    
    return {
        "test_failure_report": "", 
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True, 
        "error_message": None
    }