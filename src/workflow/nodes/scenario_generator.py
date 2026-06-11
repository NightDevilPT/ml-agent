"""
Scenario Generator Node — Phase 4: Automated Behavioral Testing Scaffolding

Acts as an automated QA Engineer to design behavioral edge-case validation suites.
It reviews feature schemas, looks at a small sample snippet from the temporary
datasets, and analyzes selected algorithm targets to generate:
  1. Adversarial mock records (outliers, corrupt types, missing values).
  2. Targeted personas to assert expected model alignment behavior.
  3. Node-level token tracking mapped as {"scenario_generator": token_value}.
"""

import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field

# Local module imports
from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage
from utils.grab_data import get_memory_safe_sample  # 👈 IMPORTED HERE

log = get_logger("scenario_generator")


# ================================================================
# Pydantic Models for Structured LLM Output
# ================================================================

class SingleTestScenario(BaseModel):
    scenario_name: str = Field(
        description="Descriptive identifier string, e.g., Extreme Outlier High, Missing Categorical, Corrupt Numeric Bound"
    )
    scenario_type: str = Field(
        description="Category classification: Outlier, Missing Data, Boundary Case, or Persona Row"
    )
    input_features_matrix: Dict[str, Any] = Field(
        description="A key-value dictionary mapping column names to mock testing input feature row data."
    )
    expected_behavioral_assertion: str = Field(
        description="An engineering description detailing how the model output is expected to respond logically to this input scenario."
    )

class ScenarioGeneratorOutput(BaseModel):
    behavioral_test_suite: List[SingleTestScenario] = Field(
        default_factory=list,
        description="An array of automated edge-case mock validation tests tailored for this specific dataset profile."
    )


# ================================================================
# Main Node Entrypoint
# ================================================================

def scenario_generator_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main LangGraph Node function.
    Leverages native structured tool bindings and reads a small safe data window
    to construct robust adversarial validation matrices.
    """
    node_name = "scenario_generator"
    log.start("Scenario Generator Node — Phase 4: Automated QA Test Synthesis")
    
    all_files = state.get("all_files", [])
    dataset_metadata = state.get("dataset_metadata", {})
    selected_config = state.get("selected_algorithm_config", {})
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    
    if not all_files or not dataset_metadata or not selected_config:
        log.error("Missing required dataset paths, metadata, or strategy choices in state context.")
        return {
            "execution_success": False,
            "error_message": "State is missing properties required for test layout generation. Run prior nodes first."
        }

    primary_file_str = all_files[0]
    
    # ----------------------------------------------------------------
    # Step 1: Secure Data Samples using grab_data Utility Module
    # ----------------------------------------------------------------
    try:
        # 👈 CLEAN REUSED EXTERNALIZED SAMPLING
        _, data_sample_string = get_memory_safe_sample(primary_file_str, sample_size=10)
        
    except Exception as read_err:
        log.error("Data profiling failed to load data records safely inside QA layer: %s", str(read_err))
        return {
            "execution_success": False,
            "error_message": f"Failed reading temporary dataset matrix sample for QA: {str(read_err)}"
        }

    # ----------------------------------------------------------------
    # Step 2: Leverage Native Structured Tool Outputs
    # ----------------------------------------------------------------
    log.section("Invoking structured QA scenario synthesizer")

    prompt = f"""Role: Automated Quality Assurance Engineer for Machine Learning Systems.
Task: Design an adversarial behavioral validation test suite containing 3 to 5 realistic mock row data profiles targeting extreme operational boundary states.

Selected Modeling Core Strategy:
- Algorithm Selected: {selected_config.get('algorithm_name')}
- Package Framework: {selected_config.get('package')}
- Core ML Task: {selected_config.get('ml_task')}

Ingested Table Structural Metadata Profile:
{json.dumps(dataset_metadata, indent=2)}

Data Snippet Context (10 Random Records with Headers):
{data_sample_string}

Task Instructions:
1. Review the parsed columns, inferred data types, and sample data records to understand true value patterns.
2. Synthesize distinct mock testing row scenarios matching the feature names exactly.
3. CRITICAL RULE: Each mock test matrix input dictionary must ONLY map column names to feature inputs. Do NOT inject inline training target labels or outcomes directly inside the feature records.
4. Target critical edge cases: missing values, continuous column outliers, data type corruption behavior, and distinct extreme user personas based on the variance seen in the records.
5. Provide precise, actionable expected behavioral descriptions to evaluate downstream code metrics safely."""

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ScenarioGeneratorOutput, include_raw=True)
    
    try:
        response = structured_llm.invoke(prompt)
        parsed_output: ScenarioGeneratorOutput = response["parsed"]
        node_token_count = extract_token_usage(response["raw"])
        
        log.info("Successfully synthesized %d localized behavioral edge-case tests", len(parsed_output.behavioral_test_suite))
        
    except Exception as llm_err:
        log.error("Failed parsing valid structured JSON matrix from AI engine: %s", str(llm_err))
        return {
            "execution_success": False,
            "error_message": f"LLM scenario structured output validation error: {str(llm_err)}"
        }

    log.info("Node complete. Total tokens for '%s': %d", node_name, node_token_count)
    log.end("Automated test generation tracking requirements finalized cleanly")

    updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
    
    return {
        "test_scenarios": [item.model_dump() for item in parsed_output.behavioral_test_suite],
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True,
        "error_message": None
    }