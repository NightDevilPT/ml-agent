"""ML Agent centralized state framework definition layout."""

from typing import TypedDict, Optional, List, Dict, Any

class MLState(TypedDict):
    """Main state object managing data trajectories across decoupled nodes."""
    
    # ----------------------------------------------------------------
    # Host Environment Inputs & Cloned Workspace Directories
    # ----------------------------------------------------------------
    target_path: str                 # Original directory folder provided by host prompt [cite: 118]
    clone_workspace: str             # Isolated workspace path, e.g., .../.temp/ml_agent_6279f94e [cite: 119]
    all_files: List[str]             # Absolute paths of raw files copied into the datasets/ folder [cite: 4]
    
    # Clean split dataset file paths generated inside processed-datasets/
    train_path: str                  # Path to processed-datasets/train-dataset.{extension}
    test_path: str                   # Path to processed-datasets/test-dataset.{extension}
    
    # ----------------------------------------------------------------
    # Target Suggestion Metadata (Step 1 of Selection Phase)
    # ----------------------------------------------------------------
    # Array of objects: [{"target_name": "...", "description": "..."}]
    target_recommendations: List[Dict[str, str]] 
    chosen_target: Optional[str]     # Selected target column name confirmed by user (y)
    
    # ----------------------------------------------------------------
    # Algorithm Selection Metadata (Step 2 of Selection Phase)
    # ----------------------------------------------------------------
    problem_type: Optional[str]      # Inferred task: "Classification" or "Regression"
    
    # Array of objects: [{"algorithm_name": "...", "weight": 0.95, "reasoning": "..."}]
    algorithm_recommendations: List[Dict[str, Any]] 
    chosen_algorithm: Optional[str]  # Selected model architecture confirmed by user (e.g., "XGBoost Classifier")
    
    # ----------------------------------------------------------------
    # Local Self-Healing Loop Feedbacks
    # ----------------------------------------------------------------
    is_data_valid: bool              # Validation flag populated by the AI Validator Node (True/False)
    consolidation_feedback: Optional[str] # Holds traceback logs or errors if cleaner/combiner nodes fail
    retry_counters: Dict[str, int]   # Keeps track of local loop iterations, e.g., {"ingestion_loop": 0}
    
    # ----------------------------------------------------------------
    # Token Tracking Operations
    # ----------------------------------------------------------------
    token_count: int                 # Global continuous cumulative token burn tracker [cite: 5]
    node_tokens: Dict[str, int]      # Local dictionary mapping node keys to token values [cite: 119]