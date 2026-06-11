"""ML Agent state definition."""

from typing import TypedDict, Optional, List, Dict, Any

class MLState(TypedDict):
    """Main state object for ML Agent workflow."""
    
    # User inputs & Paths
    target_path: str                 # Ingests raw host input string, then overriden to point to clones
    clone_workspace: str             # Contains the proper temp workspace path: C:\Users\Pawan\Desktop\FullStackProject\ml-agent\.temp\ml_agent_59945038
    
    # Dataset info (set by dataset_validator tracking the cloned files)
    dataset_metadata: Optional[Dict[str, Any]]
    all_files: List[str]             # Absolute paths of the duplicated files inside the temp workspace datasets folder
    file_count: int                  # Total count of validated data files
    
    # Algorithm choices (set by analyze_algorithm & select_algorithm)
    algorithm_recommendations: Optional[List[Dict[str, Any]]]  # List of scored candidate models
    selected_algorithm_config: Optional[Dict[str, Any]]       # Final strategy choice locked in by the user from HITL
    
    # Automated QA Scenarios & Code (set by scenario_generator & ml_architect)
    test_scenarios: Optional[List[Dict[str, Any]]]             # Stores the generated edge-case suites
    
    # Execution Tracking
    execution_success: bool          # System lifecycle tracking flag
    error_message: Optional[str]     # Tracks localized framework crash string descriptions
    
    # Token Metrics Systems
    token_count: int                 # Global cumulative total of all tokens consumed across nodes
    node_tokens: Dict[str, int]      # Maps specific nodes to their individual token usage: {[nodeName]: token value}