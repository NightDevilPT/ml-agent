"""
Analyze Algorithm Node — Phase 3A: Model Recommendation Scoring Engine
Reads preprocessed numerical data assets to recommend optimal model selection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage
from utils.grab_data import get_memory_safe_sample  

log = get_logger("analyze_algorithm")

class SingleAlgorithmRecommendation(BaseModel):
    algorithm_name: str = Field(description="Exact class name matching standard modules.")
    package: str = Field(description="Source package constraint: scikit-learn or xgboost")
    ml_task: str = Field(description="Evaluated workflow task type")
    selection_score: int = Field(description="Weighted suitability ranking score from 1 to 100")
    is_primary_recommendation: bool = Field(description="True ONLY for the single best model containing the heaviest score")
    justification: str = Field(description="Mathematically grounded reasoning statements")
    suggested_base_parameters: Dict[str, Any] = Field(default_factory=dict, description="Baseline execution configurations")

class AlgorithmSelectorOutput(BaseModel):
    algorithm_recommendations: List[SingleAlgorithmRecommendation] = Field(default_factory=list)

def analyze_algorithm_run(state: Dict[str, Any]) -> Dict[str, Any]:
    node_name = "analyze_algorithm"
    log.start("Analyze Algorithm Node — Phase 3A: Model Recommendation Scoring")
    
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    
    # 🌟 DYNAMIC PATHING: Build directory straight from clone_workspace
    clone_workspace = state.get("clone_workspace", "")
    if not clone_workspace:
        log.error("Missing clone_workspace environment path context.")
        return {
            "execution_success": False,
            "error_message": "Workspace framework directory context is missing from state."
        }
        
    processed_dir = Path(clone_workspace) / "processed_datasets"
    
    # Gather all processed CSV/Excel files directly from the directory
    processed_files = []
    if processed_dir.exists():
        processed_files = [str(f.absolute()) for f in processed_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.csv', '.xlsx', '.xls']]

    if not processed_files:
        log.error("No valid files discovered inside processed target directory: %s", str(processed_dir))
        return {
            "execution_success": False,
            "error_message": f"Preprocessed numeric datasets missing from folder: {processed_dir.name}"
        }
    
    # Target the primary preprocessed file matrix
    primary_file_str = processed_files[0]
    log.info("Analyzing clean numerical file profile at disk target: %s", Path(primary_file_str).name)
    
    try:
        columns_list, data_sample_string = get_memory_safe_sample(primary_file_str, sample_size=10)
    except Exception as read_err:
        log.error("Data profiling failed to load preprocessed records safely: %s", str(read_err))
        return {
            "execution_success": False,
            "error_message": f"Failed reading preprocessed dataset matrix sample: {str(read_err)}"
        }

    log.section("Invoking structured model recommendation analyzer")

    prompt = f"""Role: Data Science Architect.
Task: Analyze the following preprocessed, numeric data snippet to recommend optimal machine learning modeling tools.

Available Framework Options:
- scikit-learn
- xgboost

Columns Present (Preprocessed to Numeric): {columns_list}
Data Snippet Sample:
{data_sample_string}

Task Instructions:
1. Evaluate whether this dataset suggests a Classification task, Regression task, or Clustering problem.
2. Select 2 or 3 specific algorithms from the available packages that fit best.
3. Assign an explicit numerical 'selection_score' (1-100) to each.
4. Mark the winning model as 'is_primary_recommendation': true."""

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