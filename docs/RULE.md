# Development Rule Book: Agentic AutoML System

## 1. Directory & Architectural Conventions

All code adjustments must respect the workspace module layout. No loose scripts or rogue utilities may be added outside this dedicated directory structure:

```text
└── ml-agent/
    ├── datasets/                 # Local testing source directories
    ├── docs/
    │   └── RULE.md               # This rule book
    ├── pyproject.toml            # Dependency tracking (managed via uv)
    ├── src/
    │   ├── main.py               # Application entry point
    │   ├── utils/                # Core Utility Framework (DO NOT ALTER)
    │   │   ├── __init__.py
    │   │   ├── grab_data.py      # Memory-safe dataset sampling
    │   │   ├── hitl.py           # Human-in-the-loop prompts
    │   │   ├── llm.py            # Multi-provider LLM client
    │   │   ├── logger.py         # Rich terminal logging
    │   │   └── sandbox.py        # Docker isolation layer
    │   └── workflow/             # LangGraph Orchestration Layer
    │       ├── graph.py          # Node compilation and routing logic
    │       ├── state.py          # Centralized System Stateful Schema
    │       └── subgraphs/        # Sub-Graph Functional Topologies
    │           ├── {subgraph_name}_subgraph.py  # {Subgraph router}
    │           └── nodes/        # Isolated Sub-Graph Python Modules
    └── uv.lock

```

---

## 2. Core Utility Reference Manual

You must leverage the pre-built `utils` module via explicit absolute imports (`from utils.xyz import ...`). Do not alter or bypass these utility modules.

### A. Terminal Diagnostics (`utils.logger`)

**Import Pattern:**

```python
from utils.logger import get_logger

```

**Usage:** Instantiate local instances inside modules using a unique matching identification string:

```python
log = get_logger("node_name_here")

```

| Method | Purpose |
| --- | --- |
| `log.section("Title")` | Visually separates significant standalone phases in the runtime |
| `log.info("Msg %s", var)` | Logs internal parameter traces and operational updates |
| `log.warn("Message")` | Records non-fatal structural anomalies or missing fallbacks |
| `log.error("Message")` | Captures major faults that block successful data execution |

### B. Interactive Control Gates (`utils.hitl`)

**Import Pattern:**

```python
from utils.hitl import ask_human

```

**Signature Summary:**

* **Function:** `ask_human(options, title, description, default, style)`
* **Parameters:** Accepts an options dictionary matching string selection keys to descriptions, a menu header title, an optional explanation layout string, an optional default fall-back choice, and a visual diagnostic style state.
* **Returns:** `str` (The exact key chosen by the operator).

**Rule:** Use at Human Interaction Gates to capture parameter confirmations (such as target columns or models).

### C. Large Language Model API (`utils.llm`)

**Import Pattern:**

```python
from utils.llm import get_llm, extract_token_usage

```

**Signature Summary:**

* **Function:** `get_llm(provider, context, temperature)` -> Returns a standardized Chat Model interface instance.
* **Function:** `extract_token_usage(response)` -> Returns an integer tracking total token depletion count.

**Usage Rules:**

* Set deterministic, low-randomness parameter bounds (`temperature=0.1` or `0.0`) for code or structured data tasks.
* Always extract, accumulate, and update prompt metrics within the local state dictionary payload after every invocation.

### D. Isolated Safe Environments (`utils.sandbox`)

**Import Pattern:**

```python
from utils.sandbox import MlSandbox

```

**Signature Summary:**

* **Methods:**
* `.prepare(dataset_files, code_string, packages)` -> Returns a success boolean after provisioning text assets.
* `.execute()` -> Returns a `SandboxResult` containing execution logs and error parameters.
* `.cleanup()` -> Deletes runtime traces.



**Rule:** Run arbitrary runtime executions inside an isolated context manager block to prevent lingering system locks.

### E. Memory-Safe Data Sampling (`utils.grab_data`)

**Import Pattern:**

```python
from utils.grab_data import get_memory_safe_sample

```

**Signature Details:**

* **Function:** `get_memory_safe_sample(file_path, sample_size=10, window_size=1000, random_state=42)`
* **Parameters:**
* `file_path` (`Union[str, Path]`): Target system path to the CSV or Excel source table.
* `sample_size` (`int`): Maximum number of records to return within the target sample preview.
* `window_size` (`int`): Maximum number of initial data rows to read into memory to protect host RAM boundaries.
* `random_state` (`int`): Random seed value used to guarantee repeatable row sampling.


* **Returns:** `Tuple[List[str], str]`
* `List[str]`: An explicit list containing all discovered column header strings.
* `str`: A clean, formatted text matrix view containing the random sample records.



**Usage Rule:** Always use this helper tool when parsing or previewing untrusted spreadsheet sizes. Never load full datasets into memory via unconstrained pandas commands outside of an isolated worker sandbox.

---

## 3. LangGraph Workflow Sub-Graph Extension Rules

When building or updating modular steps within an orchestration sub-graph, use this 4-step workflow:

### Step 1: Update State Keys (`src/workflow/state.py`)

Declare new variables within the global centralized layout. Loose keys are strictly prohibited.

**Unified Central Schema Blueprint:**

```python
from typing import TypedDict, Optional, List, Dict, Any

class MLState(TypedDict):
    # Host Environment Inputs & Cloned Workspace Directories
    target_path: str                 # Original target folder path provided by prompt
    clone_workspace: str             # Isolated workspace path (.temp/ml_agent_dataset...)
    all_files: List[str]             # Absolute file paths of copied raw tables
    
    # Clean split data paths written after validation
    train_path: str                  # Path to processed-datasets/train-dataset.csv
    test_path: str                   # Path to processed-datasets/test-dataset.csv
    
    # Target Suggestion Metadata
    target_recommendations: List[Dict[str, str]] # [{"target_name": "...", "description": "..."}]
    chosen_target: Optional[str]     # Prediction target verified via HITL choice prompt
    
    # Algorithm Selection Metadata
    problem_type: Optional[str]      # "Classification" or "Regression"
    algorithm_recommendations: List[Dict[str, Any]] # [{"algorithm_name": "...", "weight": 0.95}]
    chosen_algorithm: Optional[str]  # Confirmed model selection verified via HITL prompt
    
    # Structural Safety & Feedback Flags
    is_data_valid: bool              # Ingestion integrity verification flag (True/False)
    consolidation_feedback: Optional[str] # Logs tracebacks if cleaner or combiner scripts fail
    retry_counters: Dict[str, int]   # Iteration limit boundaries, e.g., {"ingestion_loop": 0}
    
    # Metrics
    token_count: int                 # Global cumulative token burn tracker
    node_tokens: Dict[str, int]      # Token usage map tracking expenditure per node key

```

### Step 2: Build Isolated Node Logic (`src/workflow/subgraphs/nodes/`)

Every pipeline operation must be written as an isolated, single-purpose Python function inside its own file.

**Mandatory Sub-Graph Node Template:**

```python
"""Module header documentation block — Defines objective properties."""

from typing import Dict, Any
from workflow.state import MLState
from utils.logger import get_logger

log = get_logger("example_node_step")

def example_node_step_run(state: MLState) -> Dict[str, Any]:
    """Main processing loop operation for the node step."""
    log.section("Executing Specific Sub-Graph Function Logic")
    
    # 1. State Extraction with safe guard checking
    workspace = state.get("clone_workspace")
    if not workspace:
        log.error("Execution failed: required 'clone_workspace' path entry missing from state.")
        return {"is_data_valid": False, "consolidation_feedback": "Missing workspace link."}
        
    try:
        # 2. Add explicit functional core logic here
        pass
    except Exception as runtime_fault:
        log.error("Process aborted: %s", str(runtime_fault))
        return {"is_data_valid": False, "consolidation_feedback": str(runtime_fault)}
        
    # 3. Return only changed key elements to merge back into state cleanly
    return {
        "is_data_valid": True,
        "consolidation_feedback": None
    }

```

### Step 3: Register in Sub-Graph Router (`src/workflow/subgraphs/data_analytics_subgraph.py`)

Import and attach your completed node directly into your modular sub-graph configuration flow block.

```python
from workflow.subgraphs.nodes.your_node import your_node_run

# Inside build_analytics_subgraph():
sub_workflow.add_node("your_node", your_node_run)

```

### Step 4: Register in Main Orchestrator (`src/workflow/graph.py`)

Ensure your macro parent graph exposes the sub-graph cleanly, using edge transitions to handle downstream execution tasks or circuit-breaker halts.

---

## 4. Absolute Enforcement Mandates

### Prohibitions

* **NO LOOSE HOST OPERATIONS:** Never invoke raw Python `exec()`, `eval()`, or raw `subprocess.run()` calls on generated code inside the host system environment. Use the containerized `MlSandbox` interface instead.
* **NO RAW COPIES:** Never load large datasets into memory using unconstrained pandas calls. Use the memory-safe extraction bounds provided by `get_memory_safe_sample()`.
* **NO PRINT STATEMENTS:** Use structured logging methods (`log.info()`, `log.warn()`, `log.error()`) instead of unformatted `print()` statements.
* **NO PROMPT CLUTTER:** Do not include raw markdown code tags inside system instructions when prompting LLMs for executable script outputs.

### Mandatory Requirements

* Every single LLM execution trace must extract its token consumption metrics using `extract_token_usage()` and increment both `token_count` and `node_tokens[node_key]` before exiting.
* Every node layout block must wrap its execution routines inside a safe `try/except` handler block. If a failure occurs, it must write the trace details to `consolidation_feedback` instead of raising an unhandled host exception.