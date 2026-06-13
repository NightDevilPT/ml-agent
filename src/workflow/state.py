"""ML Agent centralized state framework definition layout."""

from typing import TypedDict, Optional, List, Dict, Any

class MLState(TypedDict):
    """Main state object managing data trajectories across decoupled nodes."""
    
    # ----------------------------------------------------------------
    # Host Environment Inputs & Cloned Workspace Directories
    # ----------------------------------------------------------------
    target_path: str                 # Original directory folder provided by host prompt
    clone_workspace: str             # Isolated workspace path, e.g., .../.temp/ml_agent_6279f94e
    
    # ----------------------------------------------------------------
    # Dataset Files & Structural Schema Metadata
    # ----------------------------------------------------------------
    dataset_metadata: Optional[Dict[str, Any]]
    all_files: List[str]             # Original copied source files inside temporary folder
    file_count: int                  # Count verification tracking field
    processed_files: List[str]       # Index 0: processed_dataset.csv (70%), Index 1: test_dataset.csv (30%)
    
    # 🌟 NEW ELEMENT ADDED HERE:
    target_variable: Optional[str]   # Explicit targeted optimization goal name (e.g., 'Close')
    
    # ----------------------------------------------------------------
    # Algorithmic Recommendation Architectures & Strategies
    # ----------------------------------------------------------------
    algorithm_recommendations: Optional[List[Dict[str, Any]]]
    selected_algorithm_config: Optional[Dict[str, Any]]       # Config locked down by user via HITL
    
    # ----------------------------------------------------------------
    # Testing Scenarios & Behavioral QA Metrics
    # ----------------------------------------------------------------
    test_scenarios: Optional[List[Dict[str, Any]]]             # Generated adversarial validation blocks
    
    # ----------------------------------------------------------------
    # Execution Tracking & Self-Healing Telemetry Loops
    # ----------------------------------------------------------------
    execution_success: bool          # Circuit-breaker flag tracking step completions
    error_message: Optional[str]     # Tracks string tracebacks when an execution falls over
    iteration_count: int             # Runs safety threshold loops (Circuit ceiling caps at 1)
    latest_sandbox_logs: str         # Stdout dump capture from container failures
    test_failure_report: str         # Grading issues returned back to script composer node
    
    # ----------------------------------------------------------------
    # Token Tracking Operations
    # ----------------------------------------------------------------
    token_count: int                 # Global continuous cumulative token burn tracker
    node_tokens: Dict[str, int]      # Local dictionary mapping node keys to token values