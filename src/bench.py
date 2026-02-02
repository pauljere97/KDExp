"""
Benchmarking utilities for model efficiency metrics.

This module provides tools for measuring:
- Latency (ms per token)
- Throughput (tokens per second)
- Memory usage (peak RAM and MPS memory)
- Model size on disk

Corresponds to thesis Section 3.9 - Efficiency Metrics.
"""

import os
import gc
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging

import torch
import psutil
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    variant: str
    params_millions: float
    model_size_gb: float
    # Latency metrics
    latency_ms_per_token: float
    latency_std: float
    # Throughput metrics
    throughput_tokens_per_sec: float
    throughput_std: float
    # Memory metrics
    peak_ram_gb: float
    peak_device_memory_gb: float
    # Metadata
    device: str
    batch_size: int
    sequence_length: int
    num_runs: int


def get_model_size_gb(model: PreTrainedModel) -> float:
    """
    Calculate model size in GB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in gigabytes
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_bytes = param_size + buffer_size
    return size_bytes / (1024 ** 3)


def get_model_params_millions(model: PreTrainedModel) -> float:
    """
    Count model parameters in millions.
    
    Args:
        model: PyTorch model
        
    Returns:
        Parameter count in millions
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def get_peak_memory_gb(device: str = "cpu") -> Dict[str, float]:
    """
    Get peak memory usage.
    
    Args:
        device: Device type (cpu, mps, cuda)
        
    Returns:
        Dictionary with memory metrics
    """
    result = {}
    
    # RAM usage
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    result["peak_ram_gb"] = ram_gb
    
    # Device-specific memory
    if device == "mps" and torch.backends.mps.is_available():
        try:
            # MPS memory tracking (available in newer PyTorch)
            if hasattr(torch.mps, 'driver_allocated_memory'):
                mps_bytes = torch.mps.driver_allocated_memory()
                result["peak_device_memory_gb"] = mps_bytes / (1024 ** 3)
            else:
                result["peak_device_memory_gb"] = 0.0
        except Exception:
            result["peak_device_memory_gb"] = 0.0
            
    elif device == "cuda" and torch.cuda.is_available():
        cuda_bytes = torch.cuda.max_memory_allocated()
        result["peak_device_memory_gb"] = cuda_bytes / (1024 ** 3)
        
    else:
        result["peak_device_memory_gb"] = 0.0
    
    return result


def clear_memory(device: str = "cpu"):
    """Clear memory caches."""
    gc.collect()
    
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class LatencyBenchmark:
    """
    Benchmark model latency (time per token).
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
        warmup_runs: int = 3,
        measurement_runs: int = 10
    ):
        """
        Initialize latency benchmark.
        
        Args:
            model: Model to benchmark
            tokenizer: Tokenizer
            device: Device to run on
            warmup_runs: Number of warmup runs
            measurement_runs: Number of measurement runs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        
        self.model.to(device)
        self.model.eval()
    
    def _generate_sample(
        self,
        prompt: str,
        max_new_tokens: int = 20
    ) -> Tuple[float, int]:
        """
        Generate tokens and measure time.
        
        Returns:
            Tuple of (total_time_seconds, num_tokens_generated)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Synchronize before timing
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Synchronize after generation
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        tokens_generated = outputs.shape[1] - input_length
        
        return total_time, tokens_generated
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 20
    ) -> Dict[str, float]:
        """
        Run latency benchmark.
        
        Args:
            prompts: List of prompts to generate from
            max_new_tokens: Tokens to generate per prompt
            
        Returns:
            Dictionary with latency metrics
        """
        clear_memory(self.device)
        
        # Warmup
        logger.info("Running warmup...")
        for i in range(min(self.warmup_runs, len(prompts))):
            self._generate_sample(prompts[i % len(prompts)], max_new_tokens)
        
        # Measurement
        logger.info("Running measurements...")
        latencies = []
        
        for i in range(self.measurement_runs):
            prompt = prompts[i % len(prompts)]
            total_time, num_tokens = self._generate_sample(prompt, max_new_tokens)
            
            if num_tokens > 0:
                ms_per_token = (total_time * 1000) / num_tokens
                latencies.append(ms_per_token)
        
        return {
            "latency_ms_per_token": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_min": np.min(latencies),
            "latency_max": np.max(latencies),
            "num_measurements": len(latencies)
        }


class ThroughputBenchmark:
    """
    Benchmark model throughput (tokens per second).
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu"
    ):
        """
        Initialize throughput benchmark.
        
        Args:
            model: Model to benchmark
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    def benchmark(
        self,
        texts: List[str],
        batch_sizes: List[int] = [1, 4, 8],
        max_length: int = 512,
        num_batches: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        Run throughput benchmark for different batch sizes.
        
        Args:
            texts: Input texts
            batch_sizes: Batch sizes to test
            max_length: Maximum sequence length
            num_batches: Number of batches to process
            
        Returns:
            Dictionary mapping batch size to metrics
        """
        results = {}
        
        for batch_size in batch_sizes:
            clear_memory(self.device)
            
            throughputs = []
            
            for batch_idx in range(num_batches):
                # Get batch
                start_idx = (batch_idx * batch_size) % len(texts)
                batch_texts = []
                for i in range(batch_size):
                    batch_texts.append(texts[(start_idx + i) % len(texts)])
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True
                ).to(self.device)
                
                total_tokens = inputs["input_ids"].numel()
                
                # Synchronize
                if self.device == "mps":
                    torch.mps.synchronize()
                elif self.device == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                # Synchronize
                if self.device == "mps":
                    torch.mps.synchronize()
                elif self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                tokens_per_sec = total_tokens / elapsed
                throughputs.append(tokens_per_sec)
            
            results[batch_size] = {
                "throughput_tokens_per_sec": np.mean(throughputs),
                "throughput_std": np.std(throughputs),
                "throughput_min": np.min(throughputs),
                "throughput_max": np.max(throughputs)
            }
        
        return results


def run_full_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    variant: str,
    prompts: List[str],
    device: str = "cpu",
    warmup_runs: int = 3,
    measurement_runs: int = 10,
    max_new_tokens: int = 20
) -> BenchmarkResult:
    """
    Run complete benchmark suite.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        model_name: Name of the model
        variant: Variant identifier (e.g., "baseline", "kd1_T4")
        prompts: Test prompts
        device: Device to run on
        warmup_runs: Warmup iterations
        measurement_runs: Measurement iterations
        max_new_tokens: Tokens to generate for latency test
        
    Returns:
        BenchmarkResult with all metrics
    """
    logger.info(f"Benchmarking {model_name} ({variant}) on {device}")
    
    model.to(device)
    model.eval()
    
    # Get model info
    params_m = get_model_params_millions(model)
    size_gb = get_model_size_gb(model)
    
    # Clear memory before benchmarks
    clear_memory(device)
    
    # Latency benchmark
    logger.info("Running latency benchmark...")
    latency_bench = LatencyBenchmark(
        model, tokenizer, device,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs
    )
    latency_results = latency_bench.benchmark(prompts, max_new_tokens)
    
    # Throughput benchmark
    logger.info("Running throughput benchmark...")
    throughput_bench = ThroughputBenchmark(model, tokenizer, device)
    throughput_results = throughput_bench.benchmark(
        prompts, batch_sizes=[1], num_batches=measurement_runs
    )
    
    # Memory measurement
    logger.info("Measuring memory...")
    memory_results = get_peak_memory_gb(device)
    
    return BenchmarkResult(
        model_name=model_name,
        variant=variant,
        params_millions=params_m,
        model_size_gb=size_gb,
        latency_ms_per_token=latency_results["latency_ms_per_token"],
        latency_std=latency_results["latency_std"],
        throughput_tokens_per_sec=throughput_results[1]["throughput_tokens_per_sec"],
        throughput_std=throughput_results[1]["throughput_std"],
        peak_ram_gb=memory_results["peak_ram_gb"],
        peak_device_memory_gb=memory_results["peak_device_memory_gb"],
        device=device,
        batch_size=1,
        sequence_length=512,
        num_runs=measurement_runs
    )


def benchmark_cpu_constrained(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_threads: int = 4,
    **kwargs
) -> BenchmarkResult:
    """
    Run CPU benchmark with constrained resources.
    
    Simulates resource-constrained deployment.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        prompts: Test prompts
        max_threads: Maximum CPU threads to use
        **kwargs: Additional arguments for run_full_benchmark
        
    Returns:
        BenchmarkResult
    """
    # Limit threads
    torch.set_num_threads(max_threads)
    
    # Force CPU
    model = model.to("cpu")
    
    result = run_full_benchmark(
        model, tokenizer,
        device="cpu",
        prompts=prompts,
        **kwargs
    )
    
    # Reset threads
    torch.set_num_threads(os.cpu_count() or 4)
    
    return result


def save_benchmark_results(
    results: List[BenchmarkResult],
    output_path: str
):
    """Save benchmark results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = [asdict(r) for r in results]
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)


def load_benchmark_results(input_path: str) -> List[BenchmarkResult]:
    """Load benchmark results from JSON."""
    with open(input_path, 'r') as f:
        results_dict = json.load(f)
    
    return [BenchmarkResult(**r) for r in results_dict]


# ============================================================================
# Quantization utilities
# ============================================================================

def check_quantization_support() -> Dict[str, bool]:
    """
    Check which quantization methods are available.
    
    Returns:
        Dictionary of method name -> available
    """
    support = {
        "torch_int8": True,  # PyTorch dynamic quantization always available
        "torch_int4": False,  # Not directly supported in PyTorch
        "bitsandbytes_int8": False,
        "bitsandbytes_int4": False
    }
    
    # Check bitsandbytes (may not work on Mac)
    try:
        import bitsandbytes
        support["bitsandbytes_int8"] = True
        support["bitsandbytes_int4"] = True
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"bitsandbytes import failed: {e}")
    
    return support


def quantize_model_dynamic(
    model: PreTrainedModel,
    dtype: torch.dtype = torch.qint8
) -> PreTrainedModel:
    """
    Apply dynamic quantization to a model.
    
    This is CPU-only quantization that works on Mac.
    
    Args:
        model: Model to quantize
        dtype: Quantization dtype (qint8)
        
    Returns:
        Quantized model
    """
    model = model.to("cpu")
    
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=dtype
    )
    
    return quantized


def try_quantize_model(
    model: PreTrainedModel,
    method: str = "int8",
    fallback_on_error: bool = True
) -> Tuple[Optional[PreTrainedModel], bool]:
    """
    Try to quantize a model with fallback handling.
    
    Args:
        model: Model to quantize
        method: Quantization method (int8, int4)
        fallback_on_error: If True, return original model on error
        
    Returns:
        Tuple of (quantized_model_or_None, success_bool)
    """
    support = check_quantization_support()
    
    try:
        if method == "int8":
            if support["torch_int8"]:
                quantized = quantize_model_dynamic(model, torch.qint8)
                return quantized, True
                
        elif method == "int4":
            if support["bitsandbytes_int4"]:
                # Would use bitsandbytes here, but may not work on Mac
                logger.warning("int4 quantization not available on this platform")
                return (model if fallback_on_error else None), False
            else:
                logger.warning("int4 quantization requires bitsandbytes")
                return (model if fallback_on_error else None), False
                
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        if fallback_on_error:
            return model, False
        return None, False
    
    return (model if fallback_on_error else None), False
