"""Script IO Writer Node.

Extracts generated raw code text blocks from the global state tracker, cleans 
markdown string formatting artifacts, and writes a runnable Python script to the workspace.
"""

import re
from pathlib import Path
from typing import Any, Dict

from workflow.state import MLState
from utils.logger import get_logger

log = get_logger("script_io_writer")


def script_io_writer_run(state: MLState) -> Dict[str, Any]:
    """Cleans the raw generated code block string and serializes it as train_executable.py."""
    log.section("Script IO Serialization Engine Initiated")

    raw_code = state.get("generated_code_script", "")
    workspace_path_str = state.get("clone_workspace", "")

    # Guard check: Ensure we have code content and a valid workspace target location
    if not raw_code:
        log.error("IO Writer Aborted: No generated code script string found in current state.")
        return {"consolidation_feedback": "IO Writer Error: Target generation script register is empty."}
        
    if not workspace_path_str:
        log.error("IO Writer Aborted: Missing 'clone_workspace' pathway marker inside state configuration.")
        return {"consolidation_feedback": "IO Writer Error: Active sandbox workspace target path is undefined."}

    workspace_path = Path(workspace_path_str)
    output_script_path = workspace_path / "train_executable.py"

    log.info("Processing raw code blocks. Stripping markdown syntax artifacts...")

    # Regex cleaning pattern: Strip away ```python or ``` code block wrappers if present
    cleaned_code = raw_code.strip()
    if cleaned_code.startswith("```"):
        # Remove leading block markers (e.g., ```python)
        cleaned_code = re.sub(r"^```[a-zA-Z]*\n", "", cleaned_code)
        # Remove trailing block markers (```)
        cleaned_code = re.sub(r"\n```$", "", cleaned_code).strip()

    # Double-check to catch any remaining edge case backtick enclosures
    cleaned_code = cleaned_code.replace("```python", "").replace("```", "").strip()

    # Append a trailing newline for clean execution styling
    cleaned_code += "\n"

    # Serialize code execution payload to the physical file system
    try:
        log.info("Writing clean executable pipeline script to disk: %s", output_script_path)
        output_script_path.write_text(cleaned_code, encoding="utf-8")
        log.info("Script successfully compiled and saved to workspace root.")
    except Exception as io_fault:
        log.error("IO Writer Disk Failure: Failed to write script out to storage channels: %s", str(io_fault))
        return {"consolidation_feedback": f"IO Writer Disk Write Exception: {str(io_fault)}"}

    # Pass control down to the next node step
    return {}