"""
Seed and reproducibility utilities.

This module provides utilities for setting random seeds and ensuring
reproducibility across experiments.
"""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, also set deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        
        # CPU
        if hasattr(torch, 'use_deterministic_algorithms') and deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Some operations don't have deterministic implementations
                pass
        
        # CUDA (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
        # MPS (Apple Silicon) - limited control but set what we can
        if torch.backends.mps.is_available():
            # MPS doesn't have as many determinism controls
            # Just ensure the main seed is set
            pass
            
    except ImportError:
        pass
    
    try:
        import transformers
        transformers.set_seed(seed)
    except (ImportError, AttributeError):
        pass


def get_generator(seed: int = 42):
    """
    Get a torch Generator with the specified seed.
    
    Args:
        seed: Random seed value
        
    Returns:
        torch.Generator instance
    """
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class SeedContext:
    """
    Context manager for temporarily setting a seed.
    
    Example:
        with SeedContext(42):
            # Code with seed 42
        # Original random state is NOT restored (use for initialization only)
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        
    def __enter__(self):
        set_seed(self.seed)
        return self
        
    def __exit__(self, *args):
        # Note: We don't restore the original state
        # This is intentional for initialization purposes
        pass


def seed_worker(worker_id: int):
    """
    Seed function for DataLoader workers.
    
    Use with DataLoader:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
