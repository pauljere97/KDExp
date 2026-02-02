"""
Knowledge Distillation Thesis Experiments

This package provides utilities for conducting knowledge distillation
experiments on LLMs, specifically designed for macOS Apple Silicon with MPS.

Modules:
- config: Configuration management
- utils_seed: Reproducibility utilities
- data_sst2: SST-2 data loading
- data_squad: SQuAD data loading
- teacher_cache: Teacher output caching
- kd_losses: KD loss functions
- trainers: Custom trainers with MPS support
- bench: Benchmarking utilities
- plots: Figure generation
- stats: Statistical analysis
- io: I/O utilities
"""

from .config import ExperimentConfig, load_config, get_config
from .utils_seed import set_seed, get_generator, seed_worker
from .kd_losses import SoftTargetLoss, SequenceKDLoss, FeatureMatchingLoss
from .bench import run_full_benchmark, BenchmarkResult

__version__ = "1.0.0"
__author__ = "Thesis Author"

__all__ = [
    "ExperimentConfig",
    "load_config", 
    "get_config",
    "set_seed",
    "get_generator",
    "seed_worker",
    "SoftTargetLoss",
    "SequenceKDLoss", 
    "FeatureMatchingLoss",
    "run_full_benchmark",
    "BenchmarkResult"
]
