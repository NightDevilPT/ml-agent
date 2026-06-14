"""AI Code Reflection and Self-Healing Remediator Node."""

from typing import Dict, Any
from utils.logger import get_logger
from workflow.state import MLState

log = get_logger("code_self_healer")


def code_self_healer_run(state: MLState) -> Dict[str, Any]:
    """Analyzes runtime error trace inputs to formulate structured correction instructions."""
    log.section("Invoking Dedicated Code Self-Healer Engine")

    runtime_stderr = state.get("runtime_stderr", "")
    train_script_code = state.get("train_script_code", "")
    evaluation_script_code = state.get("evaluation_script_code", "")
    
    retry_counters = state.get("retry_counters", {})
    generation_loops = retry_counters.get("generation_loop", 0)

    # Increment loop iteration count safely
    retry_counters["generation_loop"] = generation_loops + 1

    if not runtime_stderr:
        log.warn("Self-Healer invoked without structural stderr string markers inside state context.")
        return {"retry_counters": retry_counters}

    failing_script_context = ""
    failing_filename = "Unknown Script Target"
    
    if "train.py" in runtime_stderr or "TRAIN_SCRIPT_CODE" in runtime_stderr:
        failing_filename = "train.py (Model Training Module)"
        failing_script_context = train_script_code
    elif "main.py" in runtime_stderr or "EVALUATION_SCRIPT_CODE" in runtime_stderr:
        failing_filename = "main.py (Holdout Validation Module)"
        failing_script_context = evaluation_script_code

    log.error("Analyzing system crash. Target Identified: %s", failing_filename)

    remediation_patch_text = f"""
    [AUTOMATED REPAIR ADVISORY - ITERATION ATTEMPT #{retry_counters["generation_loop"]}]
    The previous execution cycle encountered a runtime script error.
    
    - CRASHING FILENAME TARGET: {failing_filename}
    - INTERCEPTED DESKTOP TERMINAL EXCEPTION LOG:
    {runtime_stderr}
    
    - PREVIOUS FAILING SOURCE CODE CODEBLOCK:
    {failing_script_context}
    
    INSTRUCTIONS FOR REMEDIATION:
    Identify the mismatch (e.g., column mapping, array dimension shapes, index typing, or path reference errors). 
    Recompile the scripts, ensuring that the error tracked above is thoroughly squashed while keeping all structural file configurations sound.
    """

    return {
        "consolidation_feedback": remediation_patch_text,
        "retry_counters": retry_counters
    }