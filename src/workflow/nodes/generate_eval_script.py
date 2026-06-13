"""
Evaluation Script Generator Node — Phase 8: Deployment Artifact Generation
Synthesizes a token-optimized, custom-tailored 'main.py' script utilizing memory-safe data snapshots.
"""

import json
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage
from utils.grab_data import get_memory_safe_sample

log = get_logger("generate_eval_script")

class MainEvaluatorOutput(BaseModel):
    evaluation_script_code: str = Field(
        description="Complete raw Python source code string for main.py without backticks or markdown wrap indicators."
    )


def generate_eval_script_run(state: Dict[str, Any]) -> Dict[str, Any]:
    node_name = "generate_eval_script"
    
    if state.get("execution_success") is False or state.get("test_failure_report"):
        return {
            "execution_success": False,
            "error_message": "Halted script generation: Upstream model validations did not clear successfully."
        }
        
    log.start("Evaluation Script Generator Node — Phase 8: Synthesizing Client Entrypoint")
    
    clone_workspace = state.get("clone_workspace", "")
    selected_config = state.get("selected_algorithm_config", {})
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    
    train_ref_file = Path(clone_workspace) / "processed_datasets" / "processed_dataset.csv"
    test_file = Path(clone_workspace) / "processed_datasets" / "test_dataset.csv"
    main_script_path = Path(clone_workspace) / "main.py"
    
    if not train_ref_file.exists() or not test_file.exists():
        log.error("Dataset partitions missing on disk inside the current cloned workspace context.")
        return {
            "execution_success": False,
            "error_message": "Preprocessed dataset files missing during deployment script compilation track."
        }
        
    node_token_count = 0
    try:
        columns_context, sample_data_string = get_memory_safe_sample(
            file_path=train_ref_file,
            sample_size=5,
            window_size=100
        )
        
        true_target_name = columns_context[-1]
        
        prompt = f"""Role: Expert Senior Core Machine Learning Engineer.
Task: Write a comprehensive, standalone Python evaluation script named 'main.py' to run validation checks on an entire holdout set.

Context Parameters:
- Trained Model Location: 'model.joblib'
- Training Set Location: 'processed_datasets/processed_dataset.csv'
- Holdout Testing Set Location: 'processed_datasets/test_dataset.csv'
- Framework Model Selected: {selected_config.get('algorithm_name')}
- ML Task Category: {selected_config.get('ml_task')}

Dataset Schema Context:
- Column Layout List: {json.dumps(columns_context)}
- Target Field Variable Name: '{true_target_name}' (This is explicitly the absolute last column of the matrix data structure)

Sample Records Preview Layout String:
\"\"\"
{sample_data_string}
\"\"\"

Mandatory Construction Rules:
1. Load 'model.joblib' via joblib and the entire test dataset via pandas from 'processed_datasets/test_dataset.csv'.
2. CRITICAL CORE TARGET POSITION RULE: Do NOT guess or hardcode arbitrary column names as the target variable. You must segment features and targets dynamically via index position matching:
   - Features matrix X must contain all columns except the very last one (use index slicing like df.iloc[:, :-1]).
   - Target array y_true must extract the absolute LAST column of the dataframe (use index slicing like df.iloc[:, -1]).
3. Align and type-cast the entire holdout features matrix using explicit pd.to_numeric or .astype transformations to match original training layout profiles, completely eliminating framework runtime type crashes.
4. Calculate model predictions across 100% of the rows in the test dataset. The final summary metrics MUST represent the full evaluation of all records combined.

5. MANDATORY DYNAMIC UNBOUNDED SCENARIO LOOP RULE:
   - Do NOT cap the terminal output using standard preview constraints like min(10, len(df)). You must iterate and print EVERY single record present in the test_dataset.csv matrix from index zero down to the final entry.
   - Use a sequential loop structure that executes continuously until it reaches the absolute end of the rows.

6. EXPLICIT HORIZONTAL FLAT TABLE STRUCTURE RULE:
   - Formats the output as a clean, wide, single-line horizontal table layout. Do not use multiline blocks or bullets.
   - The structural column layout order MUST be: 
     Row_Num | Open | High | Low | Close | Adj_Close | Volume | Date_year | Date_month | Date_day | Actual | Prediction | Variance
   - For every entry printed in the loop, you MUST dump all matching raw feature column value metrics alongside the prediction analytics! Print the exact feature keys inline: {json.dumps(columns_context[:-1])}.

7. 🌟 CRITICAL LINE-SPLIT & STRING SYNTAX SAFEGUARDS (PREVENT SYNTAXERRORS):
   - Never write excessively long literal string divider lines like print('=========...=========') that might break onto multiple lines during code generation.
   - Always generate divider lines programmatically using math multipliers (e.g., print('=' * 120) or print('-' * 120)).
   - Any string literals used as format template layouts should prefer explicit format templates (e.g., '{{:<12}} | {{:<12}}'.format(...) structures).
   - To prevent f-string SyntaxErrors, if a print f-string is opened with outer single quotes, any internal dictionary or template keys must use double quotes.

8. DYNAMIC COMPREHENSIVE SUMMARY STATS RULE:
   - Below the complete row scenario table grid, print a clear divider line using print('-' * 120), followed by the title: Global Performance Score Card (100 percent of Holdout Set evaluated).
   - Print rounded, clean summary metrics calculated over ALL records combined in the dataset partition.
   - For Classification, print the raw Accuracy score and a structured layout of the scikit-learn classification report.
   - For Regression, print the Mean Squared Error (MSE) and the R-Squared (R2) Variance score.
   - Footnote Rule: Add a clear plain-English explanation below each metric explaining what the score represents so a non-technical stakeholder understands it perfectly.

9. 📦 STRUCTURED RESPONSE PAYLOAD RULE:
   - Return ONLY the complete, raw Python source code string for main.py inside the 'evaluation_script_code' structured output key. 
   - Do NOT wrap the code inside markdown code block backticks or add conversational introductory prose text."""

        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(MainEvaluatorOutput, include_raw=True)
        
        response = structured_llm.invoke(prompt)
        parsed_payload: MainEvaluatorOutput = response["parsed"]
        node_token_count = extract_token_usage(response["raw"])
        
        with open(main_script_path, "w", encoding="utf-8") as f:
            f.write(parsed_payload.evaluation_script_code)
            
        log.info("AI-Scaffolded 'main.py' dynamically customized and output to workspace root layout.")
        
    except Exception as script_fault:
        log.error("AI script builder mechanism dropped a fatal runtime processing exception: %s", str(script_fault))
        return {"execution_success": False, "error_message": f"Script synthesis failed: {str(script_fault)}"}

    updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
    log.end("Deployment entrypoint script compiled successfully!")
    
    return {
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True,
        "error_message": None
    }