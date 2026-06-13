"""
ML Architect Node — Phase 5: Automated Code Generation & Script Synthesis
Generates a standalone Python training script, structures a local 'uv' 
virtual environment, and outputs a custom testing Dockerfile.
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field

from utils.llm import get_llm, extract_token_usage
from utils.logger import get_logger

log = get_logger("ml_architect")

class MlArchitectOutput(BaseModel):
    required_pip_packages: list[str] = Field(description="Pip packages required.")
    training_script_code: str = Field(description="Raw Python source code string without backticks.")


def _setup_local_uv_environment(workspace_path: Path, packages: list[str]) -> None:
    try:
        if shutil.which("uv") is None:
            log.warn("System execution tool 'uv' not detected on host. Skipping local .venv creation.")
            return

        log.info("Initializing clean local 'uv' application workspace...")
        subprocess.run(
            ["uv", "init", "--app", "--no-readme"],
            cwd=str(workspace_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        base_packages = ["pandas", "joblib", "scikit-learn", "xgboost"]
        combined_deps = list(set(packages + base_packages))
        
        log.info("Populating environment with modeling libraries: %s", combined_deps)
        subprocess.run(
            ["uv", "add"] + combined_deps,
            cwd=str(workspace_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        log.info("Successfully scaffolded localized virtual environment (.venv) inside workspace root.")
    except Exception as uv_fault:
        log.warn("Localized 'uv' setup bypassed due to an internal system fault: %s", str(uv_fault))


def _generate_manual_testing_dockerfile(workspace_path: Path, base_image: str) -> None:
    try:
        dockerfile_path = workspace_path / "Dockerfile"
        dockerfile_content = f"""# Auto-Generated Manual Testing Docker Environment
FROM {base_image}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN pip install --no-cache-dir pandas joblib scikit-learn xgboost

COPY . /workspace

# Default command runs the fast holdout evaluation testing file matrix
CMD ["python", "main.py"]
"""
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        log.info("Successfully initialized reproducible 'Dockerfile' inside workspace root.")
    except Exception as dockerfile_fault:
        log.warn("Failed creating reproducible testing Dockerfile layout: %s", str(dockerfile_fault))


def ml_architect_run(state: Dict[str, Any]) -> Dict[str, Any]:
    node_name = "ml_architect"
    log.start("ML Architect Node — Phase 5: Generating Model Training Script")
    
    global_token_count = state.get("token_count", 0)
    historical_node_tokens = state.get("node_tokens", {})
    iteration_count = state.get("iteration_count", 0) + 1
    
    if iteration_count > 1:
        log.error("Maximum self-healing attempt reached (Limit: 1). Stopping pipeline to prevent infinite looping.")
        return {
            "token_count": global_token_count,
            "node_tokens": historical_node_tokens,
            "iteration_count": iteration_count,
            "execution_success": False,
            "error_message": "Runaway self-healing loop halted: Exceeded maximum repair threshold of 1 attempt."
        }
        
    clone_workspace = state.get("clone_workspace", "")
    selected_config = state.get("selected_algorithm_config", {})
    
    processed_dir = Path(clone_workspace) / "processed_datasets"
    processed_files = [str(f.absolute()) for f in processed_dir.iterdir() if f.is_file() and f.suffix.lower() == '.csv'] if processed_dir.exists() else []
    
    if not processed_files:
        return {"execution_success": False, "error_message": "No processed numerical datasets available on disk."}
        
    # 🌟 FIX: Inside the flat container sandbox, the file resides directly in the working directory root
    target_filename = "processed_dataset.csv"
    
    latest_sandbox_logs = state.get("latest_sandbox_logs", "")
    test_failure_report = state.get("test_failure_report", "")
    
    log.section(f"Synthesizing Training Program (Attempt #{iteration_count})")
    
    healing_context = ""
    if latest_sandbox_logs:
        healing_context += f"\nCRITICAL: Your previous script crashed inside the sandbox container! Fix this exact log error:\n{latest_sandbox_logs}\n"
    if test_failure_report:
        healing_context += f"\nCRITICAL: Your previous model trained but failed behavioral validation checks:\n{test_failure_report}\nRefactor your training code to solve these case validations.\n"

    prompt = f"""Role: Expert Core Machine Learning Engineer.
Task: Write a complete, standalone Python training script named 'train.py' that loads training data, fits a model, and saves the binary.

System Configuration Context:
- Target File Dataset Name: "{target_filename}" (Sits directly in current working execution directory)
- Framework Core Package: {selected_config.get('package')}
- Target Model Architecture Class: {selected_config.get('algorithm_name')}
- Predictive ML Task Category: {selected_config.get('ml_task')}
- Hyperparameter Base Configurations: {json.dumps(selected_config.get('suggested_base_parameters', {}))}
{healing_context}

Mandatory Construction Rules:
1. Ingest training records via pandas using the exact file string name: "{target_filename}".
2. CRITICAL POSITION VARIABLE SEGMENTATION RULE: Segment your matrices using position index parameters. Do NOT assume or hardcode field label string text signatures.
   - The target validation array y_train MUST be isolated from the absolute LAST column of the matrix (use index slicing like df.iloc[:, -1]).
   - The features matrix X_train MUST contain all remaining columns except the last one (use index slicing like df.iloc[:, :-1]).
3. Split data safely using scikit-learn train_test_split (test_size=0.2, random_state=42).
4. Fit the requested model structure using the base configuration parameters and log key precision training execution parameters to stdout.
5. Save the compiled model artifact directly to the current workspace root directory as 'model.joblib' using joblib.dump().

6. CRITICAL SYNTAX & FORMATTING SAFEGUARD RULES:
   - To prevent syntax string termination crash errors, never print long, multi-line hardcoded string dividers like print('====...===='). Always generate terminal visual split lines programmatically using math operator sizing multipliers (e.g., print('=' * 80)).
   - Do NOT wrap your output string in markdown code block ticks. Return raw Python source code lines only."""

    base_image = "python:3.10-slim"
    llm = get_llm(temperature=0.0, context=8)
    structured_llm = llm.with_structured_output(MlArchitectOutput, include_raw=True)
    
    try:
        response = structured_llm.invoke(prompt)
        parsed_script_payload: MlArchitectOutput = response["parsed"]
        node_token_count = extract_token_usage(response["raw"])
        
        target_script_destination = Path(clone_workspace) / "train.py"
        with open(target_script_destination, "w", encoding="utf-8") as script_file:
            script_file.write(parsed_script_payload.training_script_code)
            
        log.info("Training script compiled successfully and saved to workspace root.")
        _setup_local_uv_environment(Path(clone_workspace), parsed_script_payload.required_pip_packages)
        _generate_manual_testing_dockerfile(Path(clone_workspace), base_image)
        
    except Exception as script_fault:
        log.error("ML Architect script syntax failure: %s", str(script_fault))
        return {"execution_success": False, "error_message": str(script_fault)}

    updated_node_tokens = {**historical_node_tokens, node_name: node_token_count}
    return {
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "iteration_count": iteration_count,
        "execution_success": True,
        "error_message": None
    }