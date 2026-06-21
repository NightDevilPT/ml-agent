# AutoML Pipeline: Detailed Node Specifications

This document provides a comprehensive technical breakdown of all operational nodes across the **Data Analytics** and **ML Architect** subgraphs of the Hierarchical Agentic AutoML System.

---

## 📊 Phase 1: Data Analytics Subgraph

### 1. `clone_dataset` (Workspace Provisioner)
* **File Location**: [`src/workflow/analytics_subgraphs/nodes/clone_dataset.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/analytics_subgraphs/nodes/clone_dataset.py)
* **Entrypoint Function**: `clone_dataset_run`
* **Operational Flow**:
  1. Validates `target_path` from the state. Checks if the path points to a file or directory, and checks for valid extensions (`.csv`, `.xlsx`, `.xls`).
  2. Generates a unique, cross-platform safe session UUID and isolated workspace folder prefix under `.temp/ml_agent_<dataset_name>_<uuid>`.
  3. Provisions subfolders: `/datasets` (cloned raw files) and `/processed-datasets` (outputs).
  4. Copies raw data files into the isolated `/datasets` workspace folder.
* **State Keys**:
  * **Inputs**: `target_path`, `token_count`, `node_tokens`
  * **Outputs**: `clone_workspace`, `all_files`, `is_data_valid` (set to `False`), `consolidation_feedback` (set to `None`), `retry_counters` (reset to `{"ingestion_loop": 0}`)

---

### 2. `combine_datasets` (Data Pooling Node)
* **File Location**: [`src/workflow/analytics_subgraphs/nodes/combine_datasets.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/analytics_subgraphs/nodes/combine_datasets.py)
* **Entrypoint Function**: `combine_datasets_run`
* **Operational Flow**:
  1. Triggered conditionally only if the input directory contains multiple spreadsheet sheets.
  2. Discovers and logs details of staged file locations, verifying multi-sheet configurations before passing variables to the cleaning node.
* **State Keys**:
  * **Inputs**: `all_files`, `token_count`, `node_tokens`
  * **Outputs**: `consolidation_feedback` (set to `None`), `node_tokens` (registers itself with 0 token overhead)

---

### 3. `single_file_cleaner` (Auto-Adaptive Cleaning Node)
* **File Location**: [`src/workflow/analytics_subgraphs/nodes/single_file_cleaner.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/analytics_subgraphs/nodes/single_file_cleaner.py)
* **Entrypoint Function**: `single_file_cleaner_run`
* **Operational Flow**:
  1. Profile the raw dataset, generating a detailed, compact markdown statistics table detailing: shapes, dtypes, null percentages, unique values, skewness, and statistical values (min, mean, standard deviation, max).
  2. Prompts the structured LLM (Gemini) to output a `PreprocessingScriptBlueprint` containing a Pydantic verified python script implementing a 9-stage cleanup process:
     - **Stage 1 (Load & Audit)**: Load raw data, print schema profiles.
     - **Stage 2 (Drop Columns)**: Exclude IDs, constants, duplicate cols, and >60% null columns.
     - **Stage 3 (Target)**: Drop missing target entries; label-encode string targets.
     - **Stage 4 (Clean Mixed-Type/Text)**: Strip strings like `$`, `%`, or `,` and convert to floats; truncate text.
     - **Stage 5 (Datetime)**: Extract year, month, day features, dropping original dates.
     - **Stage 6 (Impute)**: Impute missing numerical cells (median if skew > 1.0, otherwise mean) and categorical features (mode or 'Unknown').
     - **Stage 7 (Cap Outliers)**: Clip numerical attributes using IQR boundaries.
     - **Stage 8 (Encode)**: Apply One-Hot Encoding (for unique count <= 15) or Label Encoding (for unique count > 15) for categorical columns, casting booleans/OHE outcomes to integer metrics, saving encoding maps to `category_mappings.json`.
     - **Stage 9 (Validation Assertions)**: Verify NaN absence and type correctness using hard assertions.
  3. Automatically compiles a Docker preprocess image (`Dockerfile.preprocess`) and executes the script within an isolated container sandbox on the host, copying back `train_dataset.csv` and `category_mappings.json`.
* **State Keys**:
  * **Inputs**: `all_files`, `clone_workspace`, `token_count`, `node_tokens`
  * **Outputs**: `is_data_valid`, `train_path`, `mappings_path`, `data_process_script_code`, `target_recommendations`, `output_shape`

---

### 4. `dataset_auditor` (Dual-Gate Validation Gate)
* **File Location**: [`src/workflow/analytics_subgraphs/nodes/dataset_auditor.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/analytics_subgraphs/nodes/dataset_auditor.py)
* **Entrypoint Function**: `dataset_auditor_run`
* **Operational Flow**:
  1. **Gate 1 (Zero-Token Code Sweep)**: Inspects the cleaned data columns programmatically using pandas. Raises structural errors if unencoded object strings or residual NaNs are found.
  2. **Gate 2 (Structured Semantic Assessment)**: Prompts Gemini structured LLM to verify that categorical columns are cleanly factorized to integer/float keys and that target features mapped to 0/1 indices are correct.
  3. **Conditional Routing Loop & Circuit Breaker**: If errors are detected, it increments `ingestion_loop` and routes back to `single_file_cleaner` for self-healing. Once the retry loop hits the maximum threshold (2 attempts), the circuit breaker overrides the verification flags to prevent infinite token consumption.
* **State Keys**:
  * **Inputs**: `all_files`, `train_path`, `token_count`, `node_tokens`, `retry_counters`
  * **Outputs**: `is_data_valid`, `consolidation_feedback`, `retry_counters`

---

### 5. `splitter_export` (In-Place Partitioning Node)
* **File Location**: [`src/workflow/analytics_subgraphs/nodes/splitter_export.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/analytics_subgraphs/nodes/splitter_export.py)
* **Entrypoint Function**: `splitter_export_run`
* **Operational Flow**:
  1. Loads `train_dataset.csv` from the processed folder.
  2. Shuffles the row indices and splits the data into an **80/20** split using scikit-learn's `train_test_split` with a fixed random seed.
  3. Overwrites `train_dataset.csv` with the training slice (80%) and saves the remaining validation slice (20%) as `test_dataset.csv` for unseen model testing.
* **State Keys**:
  * **Inputs**: `clone_workspace`
  * **Outputs**: `train_path`, `test_path`, `consolidation_feedback`

---

### 6. `model_strategist` (Strategy Recommendations & HITL Ingestion Node)
* **File Location**: [`src/workflow/analytics_subgraphs/nodes/model_strategist.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/analytics_subgraphs/nodes/model_strategist.py)
* **Entrypoint Function**: `model_strategist_run`
* **Operational Flow**:
  1. Summarizes training features and distinct target values.
  2. Invokes structured LLM (Gemini) to generate rated recommendations for Target Columns, problem strategies (Classification/Regression), and model frameworks.
  3. Renders a terminal Human-in-the-Loop interactive input gateway, prompting the human developer to type choice indices (supporting single or multi-target selections).
* **State Keys**:
  * **Inputs**: `train_path`, `token_count`, `node_tokens`
  * **Outputs**: `target_recommendations`, `problem_type_recommendations`, `algorithm_recommendations`, `chosen_target`, `chosen_algorithm`, `problem_type`

---

## 🤖 Phase 2: ML Architect Subgraph

### 7. `ml_script_architect` (ML Code Generation Node)
* **File Location**: [`src/workflow/ml_subgraph/nodes/ml_script_architect.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/ml_subgraph/nodes/ml_script_architect.py)
* **Entrypoint Function**: `ml_script_architect_run`
* **Operational Flow**:
  1. Maps state attributes (problem strategy, algorithm, chosen target list) and formats a structured code instructions prompt for the Gemini LLM.
  2. Instructs the LLM to output a unified code script supporting:
     - `--mode train`: Fits estimators for all targets, caching target min/max values to a binary payload.
     - `--mode evaluate`: Loads saved models, performs prediction logic, clips regressor outputs within boundaries, scores regression (MSE, MAE, R2) and classification metrics (Accuracy, Precision, Recall, F1), and displays scorecard blocks.
     - **Mapping Inversion**: Reverts prediction and target columns back to human labels using inverted `category_mappings.json` strings with KeyError guards.
* **State Keys**:
  * **Inputs**: `clone_workspace`, `train_path`, `test_path`, `chosen_algorithm`, `chosen_target`, `problem_type`, `token_count`, `node_tokens`, `consolidation_feedback`
  * **Outputs**: `train_script_code`, `evaluation_script_code`, `workspace_readme_text`

---

### 8. `script_io_writer` (Serialization Layer Node)
* **File Location**: [`src/workflow/ml_subgraph/nodes/script_io_writer.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/ml_subgraph/nodes/script_io_writer.py)
* **Entrypoint Function**: `script_io_writer_run`
* **Operational Flow**:
  1. Cleans markdown blocks back to raw text.
  2. Writes `train.py`, `main.py`, and `README.md` to the workspace directory.
  3. Copy-harmonizes the processed test and train datasets into the workspace's `/processed-datasets` folder.
  4. Writes a production `Dockerfile` configured to pre-install dependencies (`pandas`, `joblib`, `scikit-learn`, `xgboost`), copy code modules, and chain training and evaluation steps together in a single container.
* **State Keys**:
  * **Inputs**: `clone_workspace`, `train_path`, `test_path`, `train_script_code`, `evaluation_script_code`, `workspace_readme_text`
  * **Outputs**: `script_execution_success` (in case of exceptions), `consolidation_feedback`

---

### 9. `docker_sandbox_executor` (Isolated Run Execution Node)
* **File Location**: [`src/workflow/ml_subgraph/nodes/docker_sandbox_executor.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/ml_subgraph/nodes/docker_sandbox_executor.py)
* **Entrypoint Function**: `docker_sandbox_executor_run`
* **Operational Flow**:
  1. Resolves the folder tag matching the session run ID.
  2. Builds the Docker container tag: `docker build -t <tag>:latest .`
  3. Executes the container in a transient sandbox sandbox: `docker run --rm --name <container_name> <tag>:latest`.
  4. Captures the standard output metrics log scorecard from the terminal stream, discarding extraneous logs to keep `state_record.json` extremely compact.
* **State Keys**:
  * **Inputs**: `clone_workspace`
  * **Outputs**: `script_execution_success`, `runtime_stdout`, `runtime_stderr`, `model_prediction_accurate`

---

### 10. `llm_prediction_validator` (Semantic Audit Node)
* **File Location**: [`src/workflow/ml_subgraph/nodes/llm_prediction_validator.py`](file:///c:/Users/Pawan/Desktop/FullStackProject/ml-agent/src/workflow/ml_subgraph/nodes/llm_prediction_validator.py)
* **Entrypoint Function**: `llm_prediction_validator_run`
* **Operational Flow**:
  1. Sanitizes timestamps and terminal noise from the captured stdout.
  2. Submits the scorecard snippet to the Gemini LLM for structured semantic quality checking.
  3. The LLM validates each target separately (e.g. checks binary Accuracy >= 0.55, multi-class Accuracy >= 0.45, positive R2 for regressors) and outputs a detailed critique, numeric rating, and explanation per target column.
  4. Returns the structured dictionary to the state under `model_performance_rating` mapped by target.
* **State Keys**:
  * **Inputs**: `runtime_stdout`, `chosen_target`, `chosen_algorithm`, `problem_type`, `token_count`, `node_tokens`
  * **Outputs**: `model_prediction_accurate`, `runtime_stderr`, `model_performance_rating`
