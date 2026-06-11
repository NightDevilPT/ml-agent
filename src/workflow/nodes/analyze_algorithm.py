"""
Analyze Algorithm Node — Phase 3A: Model Recommendation Scoring Engine

Uses an LLM with structured outputs to profile a data sample matrix and determine:
  1. The explicit mathematical problem space (Classification vs. Regression vs. Clustering)
  2. Optimal model choices from available production framework modules (scikit-learn, xgboost)
  3. Weighted ranking scores (1-100) where heavy weight signifies the absolute best model.
  4. Node-level token usage tracking mapped inside the state as {"analyze_algorithm": token_value}.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field

# Local module imports
from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage
from utils.grab_data import get_memory_safe_sample  # 👈 IMPORTED HERE

log = get_logger("analyze_algorithm")


# ================================================================
# Pydantic Models for Structured LLM Output
# ================================================================

class SingleAlgorithmRecommendation(BaseModel):
    algorithm_name: str = Field(
        description="Exact class name, e.g., RandomForestClassifier, XGBRegressor, LogisticRegression, KMeans"
    )
    package: str = Field(
        description="Framework source package constraint matching your environment: scikit-learn or xgboost"
    )
    ml_task: str = Field(
        description="Evaluated workflow task type: Binary Classification / Multi-Class Classification / Regression / Clustering"
    )
    selection_score: int = Field(
        description="A weighted selection score from 1 to 100 assessing suitability. Heavy weight (e.g. 95) is better than a lower weight (e.g. 60)."
    )
    is_primary_recommendation: bool = Field(
        description="Set to true ONLY for the single best model containing the heaviest selection_score."
    )
    justification: str = Field(
        description="Clear, mathematically grounded reasoning explaining why this model matches the sample columns and why it received its specific weight score."
    )
    suggested_base_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A key-value map configuring recommended baseline training variables for this estimator."
    )

class AlgorithmSelectorOutput(BaseModel):
    algorithm_recommendations: List[SingleAlgorithmRecommendation] = Field(
        default_factory=list,
        description="An array containing recommended algorithmic architectures for this specific dataset profile, sorted from heaviest weight to lowest."
    )


# ================================================================
# Main Node Entrypoint
# ================================================================

def analyze_algorithm_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main LangGraph Node function.
    Reads data assets directly out of the temporary cloned workspace path to protect source records.
    """
    node_name = "analyze_algorithm"
    log.start("Analyze Algorithm Node — Phase 3A: Model Recommendation Scoring")
    
    all_files = state.get("all_files", [])
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    
    if not all_files:
        log.error("No valid dataset files found in state tracking context.")
        return {
            "execution_success": False,
            "error_message": "Missing file trajectories. Run dataset_validator first."
        }
    
    primary_file_str = all_files[0]
    
    # ----------------------------------------------------------------
    # Step 1: Secure Data Samples using grab_data Utility Module
    # ----------------------------------------------------------------
    try:
        # 👈 CLEAN EXTERNALIZED SAMPLING
        columns_list, data_sample_string = get_memory_safe_sample(primary_file_str, sample_size=10)
        
    except Exception as read_err:
        log.error("Data profiling failed to load data records safely from temporary files: %s", str(read_err))
        return {
            "execution_success": False,
            "error_message": f"Failed reading temporary dataset matrix sample: {str(read_err)}"
        }

    # ----------------------------------------------------------------
    # Step 2: Leverage Native Structured Tool Outputs with Weighting Rules
    # ----------------------------------------------------------------
    log.section("Invoking structured model recommendation analyzer")

    prompt = f"""Role: Data Science Architect.
Task: Analyze the following data snippet (features, column structures, and sample values) to recommend optimal machine learning modeling tools.

Available Production Packages to Leverage:
- scikit-learn (Ideal for linear models, trees, ensembles, clustering, and processing)
- xgboost (Ideal for advanced gradient boosted trees on complex tabular datasets)

Columns Present: {columns_list}

Data Snippet (10 Random Records with Headers):
{data_sample_string}

Task Instructions:
1. Evaluate whether this dataset suggests a Classification task, Regression task, or Clustering problem.
2. Select 2 or 3 specific algorithms from the available packages that fit this data best.
3. Assign an explicit numerical 'selection_score' (1-100) to each algorithm as its weight. 
4. CRITICAL RULE: The heavy weight (highest score) must represent the single absolute best algorithm choice for this data pattern. Mark that specific winning model as 'is_primary_recommendation': true."""

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(AlgorithmSelectorOutput, include_raw=True)
    
    try:
        response = structured_llm.invoke(prompt)
        parsed_output: AlgorithmSelectorOutput = response["parsed"]
        node_token_count = extract_token_usage(response["raw"])
        
        sorted_recs = sorted(
            parsed_output.algorithm_recommendations, 
            key=lambda x: x.selection_score, 
            reverse=True
        )
        
        if sorted_recs:
            log.info("Winning heavy-weight algorithm selected: %s (Weight Score: %d)", 
                     sorted_recs[0].algorithm_name, sorted_recs[0].selection_score)
        
    except Exception as llm_err:
        log.error("Failed parsing valid structured JSON matrix from AI engine: %s", str(llm_err))
        return {
            "execution_success": False,
            "error_message": f"LLM structured output validation error: {str(llm_err)}"
        }

    log.info("Node complete. Total tokens for '%s': %d", node_name, node_token_count)
    log.end("Algorithm recommendation tracking complete")

    updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
    
    return {
        "algorithm_recommendations": [item.model_dump() for item in sorted_recs],
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True,
        "error_message": None
    }