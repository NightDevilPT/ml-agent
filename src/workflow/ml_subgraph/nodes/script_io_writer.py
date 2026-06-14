"""System File IO Operator Node."""

import re
import shutil
from pathlib import Path
from typing import Dict, Any

from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("script_io_writer")


def clean_markdown_code_blocks(raw_code_string: str) -> str:
    """Deterministic regex routine to strip markdown formatting backticks and decorators."""
    if not raw_code_string:
        return ""
    clean_string = re.sub(r"^```[a-zA-Z0-9]*\n", "", raw_code_string.strip())
    clean_string = re.sub(r"\n```$", "", clean_string)
    return clean_string.strip()


def script_io_writer_run(state: MLState) -> Dict[str, Any]:
    """Cleans generated tracking string payloads and serializes all system files to disk."""
    log.section("Executing Script IO File Serialization Layer")

    clone_workspace_str = state.get("clone_workspace", "").strip()
    train_path_str = state.get("train_path", "")
    test_path_str = state.get("test_path", "")
    
    train_script_code = state.get("train_script_code")
    evaluation_script_code = state.get("evaluation_script_code")
    workspace_readme_text = state.get("workspace_readme_text")

    if not clone_workspace_str or not train_path_str or not test_path_str:
        log.error("Writer Aborted: Missing path initialization parameters in workflow headers.")
        return {
            "script_execution_success": False,
            "runtime_stderr": "File System Workspace Failure: Missing path variables inside state."
        }

    workspace_root = Path(clone_workspace_str)
    target_processed_dir = workspace_root / "processed-datasets"

    try:
        if not target_processed_dir.exists():
            target_processed_dir.mkdir(parents=True, exist_ok=True)

        if train_script_code:
            clean_train = clean_markdown_code_blocks(train_script_code)
            with open(workspace_root / "train.py", "w", encoding="utf-8") as f:
                f.write(clean_train)

        if evaluation_script_code:
            clean_eval = clean_markdown_code_blocks(evaluation_script_code)
            with open(workspace_root / "main.py", "w", encoding="utf-8") as f:
                f.write(clean_eval)

        if workspace_readme_text:
            clean_readme = clean_markdown_code_blocks(workspace_readme_text)
            with open(workspace_root / "README.md", "w", encoding="utf-8") as f:
                f.write(clean_readme)

        src_train = Path(train_path_str).resolve()
        dest_train = (target_processed_dir / "train_dataset.csv").resolve()
        if src_train.exists() and src_train != dest_train:
            shutil.copy(src_train, dest_train)
            
        src_test = Path(test_path_str).resolve()
        dest_test = (target_processed_dir / "test_dataset.csv").resolve()
        if src_test.exists() and src_test != dest_test:
            shutil.copy(src_test, dest_test)

        # 🌟 THE SEQUENTIAL RUN ENGINE: Runs train and evaluate together in a single container context
        dockerfile_content = (
            "FROM python:3.12-slim\n\n"
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/\n\n"
            "ENV PYTHONDONTWRITEBYTECODE=1\n"
            "ENV PYTHONUNBUFFERED=1\n\n"
            "WORKDIR /workspace\n\n"
            "RUN uv pip install --system --no-cache pandas joblib scikit-learn xgboost\n\n"
            "COPY train.py main.py ./\n"
            "COPY processed-datasets/ ./processed-datasets/\n\n"
            "# Automatically run training first, and immediately trigger the evaluation output table\n"
            "CMD [\"bash\", \"-c\", \"python train.py --mode train && python main.py --mode evaluate\"]\n"
        )
        with open(workspace_root / "Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        log.info("Successfully scaffolded clean, chained Dockerfile execution layer.")

    except Exception as io_error:
        log.error("Fatal exception during workspace disk writing serialization: %s", str(io_error))
        return {
            "script_execution_success": False,
            "runtime_stderr": f"File System Disk IO Write Failure: {str(io_error)}"
        }

    return {"consolidation_feedback": None}