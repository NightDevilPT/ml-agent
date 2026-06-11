"""Data Ingestion and Sampling Utilities."""

from pathlib import Path
import pandas as pd
from typing import List, Tuple, Union
from utils.logger import get_logger

log = get_logger("grab_data")

def get_memory_safe_sample(
    file_path: Union[str, Path], 
    sample_size: int = 10, 
    window_size: int = 1000, 
    random_state: int = 42
) -> Tuple[List[str], str]:
    """
    Safely reads a restricted window of a dataset and extracts a random snippet string.
    Protects host environment boundaries from out-of-memory bottlenecks.
    
    Args:
        file_path: Target path to the CSV or Excel file.
        sample_size: Number of records to return in the sample.
        window_size: Number of initial rows to load into memory for sampling.
        random_state: Seed value for reproducible random sampling.
        
    Returns:
        Tuple[List[str], str]: A list of all column headers, and a string view of the sampled records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Target data file not found at: {file_path}")
        
    suffix = path.suffix.lower()
    log.info("Generating memory-safe snippet for file: %s", path.name)
    
    try:
        if suffix in ['.xlsx', '.xls']:
            # Excel files must be restricted via nrows to safeguard host RAM
            df_window = pd.read_excel(path, nrows=window_size)
            columns_list = list(df_window.columns)
            
            actual_sample_size = min(sample_size, len(df_window))
            df_sample = df_window.sample(n=actual_sample_size, random_state=random_state) if len(df_window) > 0 else df_window
        else:
            # Fast empty peek to grab headers for massive CSV footprints instantly
            df_header = pd.read_csv(path, nrows=0)
            columns_list = list(df_header.columns)
            
            # Load the guarded structural row window
            df_window = pd.read_csv(path, nrows=window_size)
            actual_sample_size = min(sample_size, len(df_window))
            df_sample = df_window.sample(n=actual_sample_size, random_state=random_state) if len(df_window) > 0 else df_window
            
        # Format the sample matrix into a clean, human-scannable layout string
        data_sample_string = df_sample.to_string(index=False)
        return columns_list, data_sample_string
        
    except Exception as err:
        log.error("Failed to generate memory-safe snippet layout: %s", str(err))
        raise RuntimeError(f"Error profiling dataset file matrix: {str(err)}")