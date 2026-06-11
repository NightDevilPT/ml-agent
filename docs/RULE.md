# Development Rule Book: Agentic AutoML System

## 1. Directory & Architectural Conventions

All code adjustments must respect the workspace module layout. No loose scripts or rogue utilities may be added outside this dedicated directory structure:

```text
└── ml-agent/
    ├── datasets/             # Local testing CSV source tables
    ├── docs/
    │   └── RULE.md           # This rule book
    ├── pyproject.toml        # Dependency tracking (managed via uv)
    ├── src/
    │   ├── main.py           # Application entry point
    │   ├── utils/            # Core Utility Framework (DO NOT ALTER)
    │   │   ├── __init__.py
    │   │   ├── grab_data.py  # Memory-safe dataset sampling
    │   │   ├── hitl.py       # Human-in-the-loop prompts
    │   │   ├── llm.py        # Multi-provider LLM client
    │   │   ├── logger.py     # Rich terminal logging
    │   │   └── sandbox.py    # Docker isolation layer
    │   └── workflow/         # LangGraph Orchestration Layer
    │       ├── graph.py      # Node compilation and routing logic
    │       ├── state.py      # Centralized System Stateful Schema
    │       └── nodes/        # Isolated Agentic Python Modules
    └── uv.lock
```

---

## 2. Core Utility Reference Manual

You must leverage the pre-built `utils` module via explicit absolute imports (`from utils.xyz import ...`). Do not re-implement or modify these features.

### A. Terminal Diagnostics (`utils.logger`)

**Import Pattern:**
```python
from utils.logger import get_logger, console
```

**Core Instance:** Always instantiate local instances within nodes using the unique module identification string:
```python
log = get_logger("node_name_here")
```

**Operational Methods:**
| Method | Purpose |
|--------|---------|
| `log.start("Message")` | Signals the beginning of a workflow node block |
| `log.info("Message %s", var)` | Tracks localized parameters and processing details |
| `log.warn("Message")` | Records non-fatal structural anomalies or missing flags |
| `log.error("Message")` | Captures execution blocks that trigger self-healing actions |
| `log.end("Message")` | Confirms clean, successful operation of a node block |
| `log.section("Title")` | Visually separates significant phases in the workflow |

### B. Interactive Control Gates (`utils.hitl`)

**Import Pattern:**
```python
from utils.hitl import ask_human
```

**Signature:**
```python
def ask_human(
    options: Dict[str, str],
    title: str = "Human Input Required",
    description: Optional[str] = None,
    default: Optional[str] = None,
    style: Literal["info", "warning", "error", "success"] = "warning"
) -> str:
```

**Usage Rule:** Must be used at the **User Interaction Gate** to let users choose between recommended algorithms. Always provide clear option descriptions and set a sensible default.

**Example:**
```python
selected_key = ask_human(
    options={"1": "Random Forest - Best for tabular data", "2": "XGBoost - Faster training"},
    title="Model Selection",
    description="Choose the algorithm for your dataset",
    default="1",
    style="info"
)
```

### C. Large Language Model API (`utils.llm`)

**Import Pattern:**
```python
from utils.llm import get_llm, extract_token_usage
```

**Signatures:**
```python
def get_llm(
    provider: Literal["docker", "gemini"] = "gemini",
    context: int = 4,
    temperature: float = 0.1
) -> ChatOpenAI | ChatGoogleGenerativeAI:
```

```python
def extract_token_usage(response: AIMessage) -> int:
```

**Usage Rules:**
- Always set deterministic low randomness (`temperature=0.0` or `0.1`) for structured JSON outputs
- Always extract and save token metadata into the centralized system state after every single invocation
- Use `with_structured_output()` with Pydantic models for reliable JSON parsing

**Example:**
```python
llm = get_llm(temperature=0.0)
structured_llm = llm.with_structured_output(MyOutputModel, include_raw=True)
response = structured_llm.invoke(prompt)
tokens_used = extract_token_usage(response["raw"])
```

### D. Isolated Safe Environments (`utils.sandbox`)

**Import Pattern:**
```python
from utils.sandbox import MlSandbox, SandboxResult
```

**Core Interface:**
| Method | Returns | Purpose |
|--------|---------|---------|
| `prepare(dataset_files, code_string, packages)` | `bool` | Copies data, writes training script |
| `execute()` | `SandboxResult` | Runs containerized training |
| `extract_model(model_filename)` | `str \| None` | Retrieves trained model |
| `cleanup()` | `bool` | Removes container and temp files |

**SandboxResult Attributes:**
- `.success: bool` - Execution status
- `.logs: str` - Complete stdout/stderr
- `.error_message: str` - Error details if failed

**Usage Rule:** Always manage resource allocations using context manager syntax to avoid lingering container locks:
```python
with MlSandbox(memory_limit="8g", timeout=600) as sandbox:
    if sandbox.prepare(dataset_files, code_string, packages):
        result = sandbox.execute()
        if result.success:
            model_path = sandbox.extract_model()
```

### E. Memory-Safe Data Sampling (`utils.grab_data`)

**Import Pattern:**
```python
from utils.grab_data import get_memory_safe_sample
```

**Signature:**
```python
def get_memory_safe_sample(
    file_path: Union[str, Path],
    sample_size: int = 10,
    window_size: int = 1000,
    random_state: int = 42
) -> Tuple[List[str], str]:
```

**Returns:**
- `List[str]`: Column headers from the dataset
- `str`: String representation of sampled records

**Usage Rule:** Always use this utility for data inspection instead of loading full datasets. Never call `pd.read_csv()` directly on large files.

**Example:**
```python
columns, sample_string = get_memory_safe_sample(
    file_path="path/to/data.csv",
    sample_size=10,
    window_size=1000
)
```

---

## 3. LangGraph Workflow Extension Rules

When implementing or extending nodes in the processing graph, you must follow this strict 3-step sequence:

### Step 1: Define or Update State Keys (`src/workflow/state.py`)

If your node introduces or requires new tracking variables, you must first register them in the centralized state schema.

**Rule:** State elements must be explicitly declared in the `MLState` TypedDict. Never pass loose dictionary keys outside the state schema.

**Current State Schema Reference:**
```python
class MLState(TypedDict):
    # Paths
    target_path: str
    clone_workspace: str
    
    # Dataset metadata
    dataset_metadata: Optional[Dict[str, Any]]
    all_files: List[str]
    file_count: int
    
    # Algorithm selection
    algorithm_recommendations: Optional[List[Dict[str, Any]]]
    selected_algorithm_config: Optional[Dict[str, Any]]
    
    # Testing & validation
    test_scenarios: Optional[List[Dict[str, Any]]]
    
    # Execution tracking
    execution_success: bool
    error_message: Optional[str]
    
    # Token metrics
    token_count: int
    node_tokens: Dict[str, int]
```

### Step 2: Build Isolated Node Logic (`src/workflow/nodes/`)

Every pipeline step must be written as an isolated Python function in its own file within the `src/workflow/nodes/` directory.

**Mandatory Node Template:**
```python
"""Node description — Phase X: What this node does."""

from typing import Any, Dict
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.llm import get_llm, extract_token_usage
from utils.grab_data import get_memory_safe_sample  # if data access needed

log = get_logger("node_name")

# Define Pydantic models for structured LLM output if applicable
class NodeOutputModel(BaseModel):
    field_name: str = Field(description="Description of field")
    # ... other fields

def node_name_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main LangGraph node function."""
    node_name = "node_name"
    log.start("Descriptive start message")
    
    # 1. Extract required state values with safe defaults
    required_value = state.get("required_key")
    if not required_value:
        log.error("Missing required state key: required_key")
        return {
            "execution_success": False,
            "error_message": "required_key missing from state"
        }
    
    # 2. Perform node-specific logic (data sampling, LLM calls, etc.)
    try:
        # LLM invocation pattern
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(NodeOutputModel, include_raw=True)
        response = structured_llm.invoke(prompt)
        parsed_output: NodeOutputModel = response["parsed"]
        node_token_count = extract_token_usage(response["raw"])
        
    except Exception as err:
        log.error("Operation failed: %s", str(err))
        return {
            "execution_success": False,
            "error_message": f"Node failed: {str(err)}"
        }
    
    # 3. Track token usage
    historical_tokens = state.get("node_tokens", {})
    updated_node_tokens = {**historical_tokens, node_name: node_token_count}
    global_token_count = state.get("token_count", 0)
    
    log.info("Node complete. Tokens consumed: %d", node_token_count)
    log.end("Node finished successfully")
    
    # 4. Return state updates (only the keys that changed)
    return {
        "output_key": parsed_output.model_dump(),
        "token_count": global_token_count + node_token_count,
        "node_tokens": updated_node_tokens,
        "execution_success": True,
        "error_message": None
    }
```

**Critical Rules for Node Development:**

| Rule | Description |
|------|-------------|
| **Error handling** | Always wrap main logic in try/except and return `execution_success: False` with descriptive `error_message` |
| **Token tracking** | Every LLM call must update `node_tokens[node_name]` and add to `token_count` |
| **State minimalism** | Return ONLY the keys that changed, not the entire state |
| **Logging completeness** | Use `log.start()` at beginning and `log.end()` at successful completion |
| **No print statements** | Use `log.info()` instead of `print()` for all terminal output |

### Step 3: Register and Route in Graph Core (`src/workflow/graph.py`)

Once a node is written, it must be wired into the processing state machine inside the graph manager file.

**Registration Pattern:**
```python
# 1. Import at top of file
from workflow.nodes.your_node import your_node_run

# 2. Inside build_graph() function
def build_graph() -> StateGraph:
    workflow = StateGraph(MLState)
    
    # Add your node
    workflow.add_node("your_node", your_node_run)
    
    # Add edges (sequential or conditional)
    workflow.add_edge("previous_node", "your_node")
    workflow.add_edge("your_node", "next_node")
    
    # For conditional routing
    workflow.add_conditional_edges(
        "your_node",
        routing_function,
        {
            "condition_a": "target_node_a",
            "condition_b": "target_node_b"
        }
    )
    
    return workflow.compile()
```

**Conditional Routing Example:**
```python
def should_skip_llm(state: MLState) -> str:
    """Return next node key based on state evaluation."""
    if state.get("file_count", 0) < 5:
        return "skip_analysis"
    return "run_analysis"

# In graph building:
workflow.add_conditional_edges(
    "dataset_validator",
    should_skip_llm,
    {
        "skip_analysis": "end",
        "run_analysis": "analyze_algorithm"
    }
)
```

---

## 4. Existing Workflow Nodes Reference

Do not modify these existing nodes without explicit permission. Use them as patterns for new nodes:

| Node File | Purpose | Key Outputs |
|-----------|---------|-------------|
| `clone_dataset.py` | Creates isolated temp workspace | `clone_workspace`, `target_path`, `all_files` |
| `dataset_validator.py` | Profiles CSV/Excel files | `dataset_metadata`, `file_count` |
| `analyze_algorithm.py` | LLM recommends models | `algorithm_recommendations`, token usage |
| `select_algorithm.py` | HITL user selection | `selected_algorithm_config` |
| `scenario_generator.py` | LLM generates test cases | `test_scenarios`, token usage |

**Current Pipeline Sequence:**
```
clone_dataset → dataset_validator → analyze_algorithm → select_algorithm → scenario_generator → END
```

---

## 5. Enforcement Mandates

### Absolute Prohibitions

| ❌ Prohibited | ✅ Correct Alternative |
|--------------|----------------------|
| `exec()` or `eval()` on generated code | Use `MlSandbox.execute()` |
| `subprocess.run()` on host | Use `MlSandbox` container execution |
| `pd.read_csv()` on large files without limits | Use `get_memory_safe_sample()` |
| `print()` statements | Use `log.info()` or `log.warn()` |
| Raising unhandled exceptions in nodes | Return `execution_success: False` with `error_message` |
| Markdown tags in LLM prompts | Use plain text or JSON instructions |
| Modifying `utils/` module files | Extend via `workflow/nodes/` only |
| Adding dependencies without `pyproject.toml` | Use `uv add <package>` |

### Mandatory Patterns

| ✅ Requirement | Where to Apply |
|---------------|----------------|
| `log.start()` and `log.end()` | Every node function |
| Token extraction after every LLM call | `extract_token_usage(response)` |
| Try/except wrapper in all nodes | Main node function body |
| State key validation before use | `state.get("key")` with existence check |
| Context manager for sandbox | `with MlSandbox() as sandbox:` |
| Pydantic models for structured LLM output | `class MyOutput(BaseModel):` |

### Error Handling Standard

```python
# Standard error return pattern
if error_condition:
    log.error("Descriptive error: %s", detail)
    return {
        "execution_success": False,
        "error_message": f"User-friendly error description: {detail}"
    }
```

### Success Return Pattern

```python
# Standard success return pattern
return {
    "key1": value1,
    "key2": value2,
    "token_count": updated_total,
    "node_tokens": updated_node_tokens,
    "execution_success": True,
    "error_message": None
}
```

---

## 6. Development Workflow Checklist

When adding a new feature or node:

- [ ] **Review RULE.md** - Understand existing patterns
- [ ] **Update `state.py`** - Add any new state keys
- [ ] **Create node file** in `workflow/nodes/` with proper naming
- [ ] **Implement node template** with logging, error handling, token tracking
- [ ] **Add node to `graph.py`** - Import, register, route
- [ ] **Test with sample dataset** from `datasets/` folder
- [ ] **Verify JSON output** in `state_record.json`
- [ ] **Check token tracking** appears in final state

---

## 7. Environment Configuration Reference

Required environment variables (set in `.env` file):

```bash
# LLM Provider (choose one)
LLM_PROVIDER=gemini  # or "docker"

# For Google Gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite

# For Docker LLM (alternative)
DOCKER_LLM_BASE_URL=http://localhost:12434/v1
DOCKER_LLM_MODEL=ai/qwen2.5
DOCKER_LLM_API_KEY=not-needed

# Docker sandbox defaults (optional)
DOCKER_MEMORY_LIMIT=8g
DOCKER_TIMEOUT=600
```
