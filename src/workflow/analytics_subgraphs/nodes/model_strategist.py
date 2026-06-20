"""Model Strategy Profiler, Recommendation, and HITL Selection Ingestion Node.

Phase: Pre-Training Strategy Selection and Human-In-The-Loop (HITL) Ingestion Boundary
Optimization Strategy: Predictive Confidence Weights Allocation Matrix with Multi-Target Capability
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd
from pydantic import BaseModel, Field

from utils.llm import extract_token_usage, get_llm
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("model_strategist")


# ==============================================================================
# Pydantic Schemas: The Strategy Recommendation Contract
# ==============================================================================
class TargetCandidate(BaseModel):
    column_name: str = Field(description="The exact name of the potential target column feature (case-sensitive).")
    reasoning: str = Field(description="A concise description explaining why a data scientist would predict this column.")
    confidence_weight: float = Field(description="Mathematical confidence score from 0.0 (lowest) to 1.0 (highest) showing how suitable this column is as a model target.")

class ProblemTypeCandidate(BaseModel):
    problem_type: str = Field(description="The problem task strategy type, either 'Classification' or 'Regression'.")
    rationale: str = Field(description="A concise description explaining why this task strategy fits the dataset patterns.")
    suitability_weight: float = Field(description="Mathematical confidence score from 0.0 to 1.0 showing how suitable this problem type is.")

class AlgorithmCandidate(BaseModel):
    algorithm_name: str = Field(description="The exact engineering name of the algorithm (e.g., 'XGBoostClassifier', 'LightGBMRegressor').")
    rationale: str = Field(description="A concise description explaining why this specific model structure fits this data matrix pattern.")
    suitability_weight: float = Field(description="Mathematical compatibility score from 0.0 to 1.0 showing how well this algorithm fits this specific dataset profile.")

class StrategyBlueprint(BaseModel):
    dataset_analysis_rationale: str = Field(
        description="A senior-level summary of the processed dataset's mathematical distribution rules."
    )
    recommended_problem_types: List[ProblemTypeCandidate] = Field(
        description="A curated list of task strategy types (Classification or Regression) suitable for predicting target candidates."
    )
    target_column_candidates: List[TargetCandidate] = Field(
        description="A list of priority column candidates that are mathematically suitable to be the ML target (y)."
    )
    recommended_algorithms: List[AlgorithmCandidate] = Field(
        description="A curated list of top-tier machine learning algorithm models well-suited for this data layout."
    )


# ==============================================================================
# Node Entry Control Point
# ==============================================================================
def model_strategist_run(state: MLState) -> Dict[str, Any]:
    """Generates weighted target and model recommendations, then collects user choices via an interactive terminal interface."""
    log.section("Model Strategy Ingestion Engine Initiated")

    train_path_str = state.get("train_path", "")
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})

    if not train_path_str:
        log.error("Strategy Engine Aborted: Missing 'train_path' parameter inside global state.")
        return {
            "consolidation_feedback": "Strategy Fault: Cannot build recommendations without an ingested training file.",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "model_strategist": 0}
        }

    train_file_path = Path(train_path_str)

    # 1. Ingest a compressed string layout profile of the clean training data
    try:
        df_train = pd.read_csv(train_file_path)
        train_rows_snippet = df_train.head(3).to_csv(index=False, sep="|")
        
        # Build compact column metadata summary
        summary_lines = ["Column | Dtype | UniqueCount"]
        for col in df_train.columns:
            summary_lines.append(f"{col} | {df_train[col].dtype} | {df_train[col].nunique()}")
        columns_summary = "\n".join(summary_lines)
    except Exception as read_fault:
        log.error("Strategy Engine IO Error: Failed reading cleaned training dataset: %s", str(read_fault))
        return {
            "consolidation_feedback": f"Strategy Engine IO Fault: {str(read_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "model_strategist": 0}
        }

    # 2. Invoke Structured LLM to recommend modeling configurations with confidence weights
    llm = get_llm(provider="gemini", temperature=0.0)
    structured_strategist = llm.with_structured_output(StrategyBlueprint, include_raw=True)

    prompt = f"""Analyze this processed numerical dataset summary:

[COLUMNS SUMMARY]
{columns_summary}

[DATA SAMPLES]
{train_rows_snippet}

Recommend:
1. Target variables (y) with confidence weights.
2. Problem types ('Classification' or 'Regression') with suitability weights.
3. Matching algorithms with suitability weights."""

    log.info("Requesting automated model training blueprint recommendations from LLM.")
    try:
        response = structured_strategist.invoke(prompt)
        blueprint: StrategyBlueprint = response["parsed"]
        node_spent = extract_token_usage(response["raw"])
        log.info("Modeling blueprint compiled successfully. Ingestion token burn: %d", node_spent)
    except Exception as ai_fault:
        log.error("Platform Fault: LLM strategist failed to emit structured blueprint: %s", str(ai_fault))
        return {
            "consolidation_feedback": f"LLM Strategist Exception: {str(ai_fault)}",
            "token_count": global_token_count,
            "node_tokens": {**historical_node_tokens, "model_strategist": 0}
        }

    log.info("AI Analysis Summary: %s", blueprint.dataset_analysis_rationale)
    
    sorted_targets = sorted(blueprint.target_column_candidates, key=lambda x: x.confidence_weight, reverse=True)
    sorted_problems = sorted(blueprint.recommended_problem_types, key=lambda x: x.suitability_weight, reverse=True)
    sorted_algos = sorted(blueprint.recommended_algorithms, key=lambda x: x.suitability_weight, reverse=True)

    target_options = [t.column_name for t in sorted_targets]
    problem_options = [p.problem_type for p in sorted_problems]
    algo_options = [a.algorithm_name for a in sorted_algos]

    # ==============================================================================
    # 🌟 STEP 3: HUMAN-IN-THE-LOOP (HITL) INTERACTIVE INGESTION INTERFACE
    # ==============================================================================
    print("\n" + "="*80)
    print("AUTOMATED ML PLATFORM: HUMAN-IN-THE-LOOP STRATEGY GATEWAY")
    print("="*80)
    
    # HITL Selection 1: Target Feature Sorted by Confidence
    print("\nDISCOVERED TARGET FEATURE CANDIDATES (Ranked by Confidence):")
    for idx, cand in enumerate(sorted_targets, 1):
        print(f"  [{idx}] Column Name: '{cand.column_name}' | Confidence Weight: {cand.confidence_weight:.2f}")
        print(f"      Description: {cand.reasoning}")
        
    chosen_target_output: Union[str, List[str]] = ""
    while True:
        try:
            user_raw_input = input(f"\nSelect Target Column Index (Separate multiple with commas, e.g., 1,2): ").strip()
            
            # Parse commas and convert selections into clear integer elements
            raw_indices = [x.strip() for x in user_raw_input.split(",")]
            selected_indices = [int(i) for i in raw_indices if i.isdigit()]
            
            # Boundary constraint assertion validation check
            if selected_indices and all(1 <= idx <= len(target_options) for idx in selected_indices):
                # Map selected list back to raw column strings
                mapped_targets = [target_options[idx - 1] for idx in selected_indices]
                
                # If the user picked only one target, store it as a plain string; otherwise, store the whole list
                chosen_target_output = mapped_targets[0] if len(mapped_targets) == 1 else mapped_targets
                break
                
            print(f"Error: Invalid Choice. Select a valid index number or sequence list from 1 to {len(target_options)}.")
        except ValueError:
            print("Error: Invalid input format. Please enter clean whole integers separated by commas (e.g., 1, 2).")

    # HITL Selection 2: Problem Type Sorted by Suitability
    print("\nRECOMMENDED PROBLEM TYPES (Ranked by Suitability):")
    for idx, cand_prob in enumerate(sorted_problems, 1):
        print(f"  [{idx}] Problem Type: '{cand_prob.problem_type}' | Suitability Weight: {cand_prob.suitability_weight:.2f}")
        print(f"      Description: {cand_prob.rationale}")
        
    chosen_problem_type = None
    while True:
        try:
            prob_selection = int(input(f"\nSelect Problem Type Index (1-{len(problem_options)}): ").strip())
            if 1 <= prob_selection <= len(problem_options):
                chosen_problem_type = problem_options[prob_selection - 1]
                break
            print(f"Error: Invalid Choice. Select a valid index number from 1 to {len(problem_options)}.")
        except ValueError:
            print("Error: Invalid input. Please enter a valid integer choice index.")

    # HITL Selection 3: Training Algorithm Sorted by Suitability
    print("\nRECOMMENDED MACHINE LEARNING ALGORITHMS (Ranked by Suitability):")
    for idx, cand_algo in enumerate(sorted_algos, 1):
        print(f"  [{idx}] Algorithm Name: {cand_algo.algorithm_name} | Suitability Weight: {cand_algo.suitability_weight:.2f}")
        print(f"      Description: {cand_algo.rationale}")
        
    chosen_algorithm = None
    while True:
        try:
            algo_selection = int(input(f"\nSelect Training Model Algorithm Index (1-{len(algo_options)}): ").strip())
            if 1 <= algo_selection <= len(algo_options):
                chosen_algorithm = algo_options[algo_selection - 1]
                break
            print(f"Error: Invalid Choice. Select a valid number from 1 to {len(algo_options)}.")
        except ValueError:
            print("Error: Invalid input. Please enter a valid integer choice index.")

    print("\n" + "="*80)
    log.info("HITL Input Ingested. Selected Target(s): %s | Problem Type: '%s' | Selected Model Architecture: '%s'", str(chosen_target_output), chosen_problem_type, chosen_algorithm)
    print("="*80 + "\n")

    # Reformat data structures into standard arrays of objects for global state tracking
    state_target_recommendations = [
        {"target_name": t.column_name, "description": t.reasoning, "weight": t.confidence_weight} 
        for t in sorted_targets
    ]
    state_problem_type_recommendations = [
        {"problem_type": p.problem_type, "description": p.rationale, "weight": p.suitability_weight}
        for p in sorted_problems
    ]
    state_algo_recommendations = [
        {"algorithm_name": a.algorithm_name, "description": a.rationale, "weight": a.suitability_weight} 
        for a in sorted_algos
    ]

    # 4. Commit structured updates back to the orchestrator state registers
    return {
        "target_recommendations": state_target_recommendations,
        "problem_type_recommendations": state_problem_type_recommendations,
        "algorithm_recommendations": state_algo_recommendations,
        "chosen_target": chosen_target_output,
        "chosen_algorithm": chosen_algorithm,
        "problem_type": chosen_problem_type,
        "token_count": global_token_count + node_spent,
        "node_tokens": {**historical_node_tokens, "model_strategist": node_spent}
    }