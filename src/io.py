"""
I/O utilities for saving and loading experiment results.

This module handles:
- CSV/JSON reading and writing
- Run registry for tracking experiments
- Result aggregation

Corresponds to thesis Chapter 4 - Results Output.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Information about an experiment run."""
    run_id: str
    task: str
    method: str
    model: str
    variant: str
    seed: int
    timestamp: str
    status: str  # "running", "completed", "failed"
    config_hash: str
    metrics: Optional[Dict[str, float]] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None


class RunRegistry:
    """
    Registry for tracking experiment runs.
    
    Provides idempotent run management - skips already completed runs.
    """
    
    def __init__(self, registry_path: str):
        """
        Initialize run registry.
        
        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: Dict[str, RunInfo] = {}
        self._load()
    
    def _load(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                for run_id, run_dict in data.items():
                    self.runs[run_id] = RunInfo(**run_dict)
    
    def _save(self):
        """Save registry to disk."""
        data = {run_id: asdict(run) for run_id, run in self.runs.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _generate_run_id(
        task: str,
        method: str,
        model: str,
        variant: str,
        seed: int
    ) -> str:
        """Generate unique run ID."""
        key = f"{task}_{method}_{model}_{variant}_{seed}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        """Hash configuration for change detection."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def register_run(
        self,
        task: str,
        method: str,
        model: str,
        variant: str,
        seed: int,
        config: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        Register a new run.
        
        Args:
            task: Task name
            method: KD method
            model: Model name
            variant: Variant identifier
            seed: Random seed
            config: Run configuration
            
        Returns:
            Tuple of (run_id, should_run) - False if already completed
        """
        run_id = self._generate_run_id(task, method, model, variant, seed)
        config_hash = self._hash_config(config)
        
        # Check if run exists and is completed with same config
        if run_id in self.runs:
            existing = self.runs[run_id]
            if existing.status == "completed" and existing.config_hash == config_hash:
                logger.info(f"Run {run_id} already completed, skipping")
                return run_id, False
        
        # Register new run
        run_info = RunInfo(
            run_id=run_id,
            task=task,
            method=method,
            model=model,
            variant=variant,
            seed=seed,
            timestamp=datetime.now().isoformat(),
            status="running",
            config_hash=config_hash
        )
        
        self.runs[run_id] = run_info
        self._save()
        
        return run_id, True
    
    def complete_run(
        self,
        run_id: str,
        metrics: Dict[str, float],
        output_path: Optional[str] = None
    ):
        """Mark run as completed with metrics."""
        if run_id in self.runs:
            self.runs[run_id].status = "completed"
            self.runs[run_id].metrics = metrics
            self.runs[run_id].output_path = output_path
            self._save()
    
    def fail_run(self, run_id: str, error_message: str):
        """Mark run as failed."""
        if run_id in self.runs:
            self.runs[run_id].status = "failed"
            self.runs[run_id].error_message = error_message
            self._save()
    
    def get_completed_runs(self, task: Optional[str] = None) -> List[RunInfo]:
        """Get all completed runs, optionally filtered by task."""
        completed = [r for r in self.runs.values() if r.status == "completed"]
        if task:
            completed = [r for r in completed if r.task == task]
        return completed
    
    def get_run_metrics(self, task: str) -> pd.DataFrame:
        """Get metrics DataFrame for all completed runs of a task."""
        runs = self.get_completed_runs(task)
        
        rows = []
        for run in runs:
            row = {
                "run_id": run.run_id,
                "task": run.task,
                "method": run.method,
                "model": run.model,
                "variant": run.variant,
                "seed": run.seed,
            }
            if run.metrics:
                row.update(run.metrics)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def clear_failed(self):
        """Remove failed runs from registry."""
        self.runs = {k: v for k, v in self.runs.items() if v.status != "failed"}
        self._save()


def save_csv(df: pd.DataFrame, path: str, **kwargs):
    """
    Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        path: Output path
        **kwargs: Additional arguments for to_csv
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info(f"Saved CSV to {path}")


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    
    Args:
        path: Input path
        **kwargs: Additional arguments for read_csv
        
    Returns:
        DataFrame
    """
    return pd.read_csv(path, **kwargs)


def save_json(data: Any, path: str, indent: int = 2):
    """
    Save data to JSON.
    
    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Any:
    """
    Load data from JSON.
    
    Args:
        path: Input path
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def aggregate_run_results(
    raw_runs_dir: str,
    output_path: str,
    task: str
) -> pd.DataFrame:
    """
    Aggregate results from multiple runs.
    
    Args:
        raw_runs_dir: Directory containing run results
        output_path: Path for aggregated output
        task: Task name
        
    Returns:
        Aggregated DataFrame
    """
    raw_dir = Path(raw_runs_dir)
    all_results = []
    
    # Find all result files for this task
    for result_file in raw_dir.glob(f"*{task}*.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            logger.warning(f"Error loading {result_file}: {e}")
    
    if not all_results:
        logger.warning(f"No results found for task {task}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Aggregate across seeds
    group_cols = [c for c in df.columns if c not in ["seed", "run_id"] and not c.startswith("metric_")]
    metric_cols = [c for c in df.columns if c.startswith("metric_") or c in ["accuracy", "f1", "exact_match"]]
    
    if group_cols and metric_cols:
        agg_dict = {col: ["mean", "std"] for col in metric_cols if col in df.columns}
        if agg_dict:
            aggregated = df.groupby(group_cols).agg(agg_dict)
            aggregated.columns = ["_".join(col).strip() for col in aggregated.columns]
            aggregated = aggregated.reset_index()
        else:
            aggregated = df
    else:
        aggregated = df
    
    save_csv(aggregated, output_path)
    
    return aggregated


def create_main_results_table(
    results_df: pd.DataFrame,
    task: str,
    output_path: str
) -> pd.DataFrame:
    """
    Create main results table for thesis.
    
    Args:
        results_df: DataFrame with all results
        task: Task name
        output_path: Output path
        
    Returns:
        Formatted results DataFrame
    """
    # Expected columns
    columns = [
        "model", "variant", "params_b",
        "task_score_primary", "task_score_secondary",
        "model_size_gb", "peak_mem_gb",
        "latency_ms_per_token", "throughput_tok_per_sec",
        "delta_vs_teacher", "delta_vs_b0"
    ]
    
    # Map existing columns
    col_mapping = {
        "accuracy": "task_score_primary",
        "f1": "task_score_secondary",
        "exact_match": "task_score_primary",
    }
    
    df = results_df.copy()
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Add missing columns with NaN
    for col in columns:
        if col not in df.columns:
            df[col] = float("nan")
    
    # Calculate deltas if possible
    if "task_score_primary" in df.columns:
        teacher_score = df[df["variant"] == "teacher"]["task_score_primary"].values
        if len(teacher_score) > 0:
            df["delta_vs_teacher"] = df["task_score_primary"] - teacher_score[0]
        
        baseline_score = df[df["variant"] == "baseline"]["task_score_primary"].values
        if len(baseline_score) > 0:
            df["delta_vs_b0"] = df["task_score_primary"] - baseline_score[0]
    
    # Select and order columns
    output_df = df[[c for c in columns if c in df.columns]]
    
    save_csv(output_df, output_path)
    
    return output_df


def create_ablation_table(
    results_df: pd.DataFrame,
    output_path: str
) -> pd.DataFrame:
    """
    Create ablation study table.
    
    Args:
        results_df: DataFrame with ablation results
        output_path: Output path
        
    Returns:
        Ablation table DataFrame
    """
    columns = ["variant", "T", "alpha", "lambda", "val_score", "test_score", "selected"]
    
    df = results_df.copy()
    
    # Ensure columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = float("nan") if col not in ["variant", "selected"] else ""
    
    # Mark best config as selected
    if "test_score" in df.columns and df["test_score"].notna().any():
        best_idx = df["test_score"].idxmax()
        df["selected"] = ""
        df.loc[best_idx, "selected"] = "✓"
    
    output_df = df[[c for c in columns if c in df.columns]]
    
    save_csv(output_df, output_path)
    
    return output_df


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return Path(path).exists()


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the latest checkpoint from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    checkpoints = list(checkpoint_path.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    
    return str(checkpoints[-1])
