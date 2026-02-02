"""
Configuration loader for Knowledge Distillation experiments.

This module provides configuration management using pydantic and YAML.
Corresponds to Chapter 3 experimental design.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class DeviceType(str, Enum):
    """Supported device types."""
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class PrecisionType(str, Enum):
    """Supported precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class DeviceConfig:
    """Device and precision configuration."""
    preference: List[str] = field(default_factory=lambda: ["mps", "cpu"])
    precision: Dict[str, str] = field(default_factory=lambda: {
        "mps": "fp32", "cpu": "fp32", "cuda": "fp16"
    })
    max_memory_gb: float = 32.0
    gradient_checkpointing: bool = True
    empty_cache_steps: int = 10


@dataclass
class TeacherConfig:
    """Teacher model configuration."""
    primary: str = ""
    local_fallback: str = ""
    auto_fallback: bool = True
    load_in_4bit: bool = False


@dataclass
class StudentConfig:
    """Student model configuration."""
    name: str = ""
    params_b: float = 0.0
    description: str = ""
    base_model: Optional[str] = None


@dataclass
class LoRAConfig:
    """LoRA/PEFT configuration."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TaskConfig:
    """Task-specific configuration."""
    enabled: bool = True
    dataset: str = ""
    subset: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    primary_metric: str = ""
    max_length: Dict[str, int] = field(default_factory=dict)
    subset_sizes: Dict[str, Dict[str, Optional[int]]] = field(default_factory=dict)
    num_labels: Optional[int] = None
    label_names: Optional[List[str]] = None
    use_validation_as_test: bool = False
    doc_stride: int = 128
    max_answer_length: int = 30


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    per_device_train_batch: Dict[str, int] = field(default_factory=lambda: {"fast": 2, "full": 4})
    per_device_eval_batch: Dict[str, int] = field(default_factory=lambda: {"fast": 4, "full": 8})
    gradient_accumulation: Dict[str, int] = field(default_factory=lambda: {"fast": 4, "full": 4})
    epochs: Dict[str, int] = field(default_factory=lambda: {"fast": 1, "full": 3})
    seeds: Dict[str, List[int]] = field(default_factory=lambda: {"fast": [42], "full": [42, 123, 456]})


@dataclass 
class KDMethodConfig:
    """Knowledge distillation method configuration."""
    enabled: bool = True
    name: str = ""
    description: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    grid_search: bool = False
    tasks: Optional[Dict[str, bool]] = None
    layer_mapping: Optional[Dict[str, Any]] = None
    memory_optimization: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkConfig:
    """Benchmarking configuration."""
    warmup_runs: int = 3
    measurement_runs: int = 10
    batch_size: int = 1
    throughput_batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    num_samples: int = 100
    cpu_benchmark_enabled: bool = True
    constrained_max_threads: int = 4


@dataclass
class OutputConfig:
    """Output paths and settings."""
    results_dir: str = "./results"
    raw_runs_dir: str = "./results/raw_runs"
    summary_dir: str = "./results/summary"
    figures_dir: str = "./results/figures"
    models_dir: str = "./results/models"
    figure_dpi: int = 300
    figure_format: str = "png"


class ExperimentConfig:
    """
    Main experiment configuration class.
    
    Loads configuration from YAML file and environment variables.
    Environment variables take precedence over YAML values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default path.
        """
        self.config_path = config_path or self._find_config_path()
        self._raw_config = self._load_yaml()
        self._apply_env_overrides()
        self._parse_config()
        
    def _find_config_path(self) -> str:
        """Find the configuration file."""
        possible_paths = [
            Path("configs/experiment.yaml"),
            Path("../configs/experiment.yaml"),
            Path(__file__).parent.parent / "configs" / "experiment.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError("Could not find experiment.yaml configuration file")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Fast mode
        env_fast_mode = os.getenv("FAST_MODE", "").lower()
        if env_fast_mode in ("true", "1", "yes"):
            self._raw_config.setdefault("experiment", {})["fast_mode"] = True
        elif env_fast_mode in ("false", "0", "no"):
            self._raw_config.setdefault("experiment", {})["fast_mode"] = False
            
        # Device
        env_device = os.getenv("DEVICE")
        if env_device:
            self._raw_config.setdefault("device", {})["preference"] = [env_device]
            
        # Models
        models = self._raw_config.setdefault("models", {})
        teacher = models.setdefault("teacher", {})
        
        if os.getenv("TEACHER_MODEL_PRIMARY"):
            teacher["primary"] = os.getenv("TEACHER_MODEL_PRIMARY")
        if os.getenv("TEACHER_MODEL_FALLBACK"):
            teacher["local_fallback"] = os.getenv("TEACHER_MODEL_FALLBACK")
            
        students = models.setdefault("students", {})
        if os.getenv("STUDENT_S1"):
            students.setdefault("s1", {})["name"] = os.getenv("STUDENT_S1")
        if os.getenv("STUDENT_S2"):
            students.setdefault("s2", {})["name"] = os.getenv("STUDENT_S2")
            
        # Seeds
        env_seeds = os.getenv("SEEDS")
        if env_seeds:
            seeds = [int(s.strip()) for s in env_seeds.split(",")]
            training = self._raw_config.setdefault("training", {})
            training.setdefault("seeds", {})["fast"] = seeds[:1]
            training.setdefault("seeds", {})["full"] = seeds
            
        # Precision
        env_precision = os.getenv("PRECISION")
        if env_precision:
            device_cfg = self._raw_config.setdefault("device", {})
            precision = device_cfg.setdefault("precision", {})
            precision["mps"] = env_precision
            precision["cpu"] = env_precision
            
        # Paths
        if os.getenv("RESULTS_DIR"):
            self._raw_config.setdefault("output", {}).setdefault("dirs", {})["results"] = os.getenv("RESULTS_DIR")
        if os.getenv("MODELS_DIR"):
            self._raw_config.setdefault("output", {}).setdefault("dirs", {})["models"] = os.getenv("MODELS_DIR")
        if os.getenv("FIGURES_DIR"):
            self._raw_config.setdefault("output", {}).setdefault("dirs", {})["figures"] = os.getenv("FIGURES_DIR")
        if os.getenv("HF_HOME"):
            os.environ["HF_HOME"] = os.getenv("HF_HOME")
            os.environ["TRANSFORMERS_CACHE"] = os.getenv("HF_HOME")

    def _parse_config(self):
        """Parse raw config into structured objects."""
        exp = self._raw_config.get("experiment", {})
        self.experiment_name = exp.get("name", "kd_thesis_experiments")
        self.fast_mode = exp.get("fast_mode", True)
        
        # Device config
        dev = self._raw_config.get("device", {})
        self.device = DeviceConfig(
            preference=dev.get("preference", ["mps", "cpu"]),
            precision=dev.get("precision", {"mps": "fp32", "cpu": "fp32"}),
            max_memory_gb=dev.get("memory", {}).get("max_memory_gb", 32),
            gradient_checkpointing=dev.get("memory", {}).get("gradient_checkpointing", True),
            empty_cache_steps=dev.get("memory", {}).get("empty_cache_steps", 10)
        )
        
        # Models
        models = self._raw_config.get("models", {})
        teacher = models.get("teacher", {})
        self.teacher = TeacherConfig(
            primary=teacher.get("primary", ""),
            local_fallback=teacher.get("local_fallback", ""),
            auto_fallback=teacher.get("auto_fallback", True),
            load_in_4bit=teacher.get("load_in_4bit", False)
        )
        
        students = models.get("students", {})
        s1 = students.get("s1", {})
        self.student_s1 = StudentConfig(
            name=s1.get("name", ""),
            params_b=s1.get("params_b", 1.1),
            description=s1.get("description", "")
        )
        
        s2 = students.get("s2", {})
        self.student_s2 = StudentConfig(
            name=s2.get("name", "quantized-proxy"),
            params_b=s2.get("params_b", 0.35),
            description=s2.get("description", ""),
            base_model=s2.get("base_model")
        )
        
        # PEFT/LoRA
        peft = self._raw_config.get("peft", {})
        lora = peft.get("lora", {}) if peft.get("enabled", True) else {}
        self.lora = LoRAConfig(
            enabled=peft.get("enabled", True),
            r=lora.get("r", 16),
            lora_alpha=lora.get("lora_alpha", 32),
            lora_dropout=lora.get("lora_dropout", 0.05),
            target_modules=lora.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            bias=lora.get("bias", "none"),
            task_type=lora.get("task_type", "CAUSAL_LM")
        )
        
        # Tasks
        tasks = self._raw_config.get("tasks", {})
        self.tasks = {}
        for task_name, task_cfg in tasks.items():
            self.tasks[task_name] = TaskConfig(
                enabled=task_cfg.get("enabled", True),
                dataset=task_cfg.get("dataset", ""),
                subset=task_cfg.get("subset"),
                metrics=task_cfg.get("metrics", []),
                primary_metric=task_cfg.get("primary_metric", ""),
                max_length=task_cfg.get("max_length", {}),
                subset_sizes=task_cfg.get("subset_sizes", {}),
                num_labels=task_cfg.get("num_labels"),
                label_names=task_cfg.get("label_names"),
                use_validation_as_test=task_cfg.get("use_validation_as_test", False),
                doc_stride=task_cfg.get("doc_stride", 128),
                max_answer_length=task_cfg.get("max_answer_length", 30)
            )
        
        # KD Methods
        kd = self._raw_config.get("kd_methods", {})
        self.kd_methods = {}
        for method_name, method_cfg in kd.items():
            self.kd_methods[method_name] = KDMethodConfig(
                enabled=method_cfg.get("enabled", True),
                name=method_cfg.get("name", method_name),
                description=method_cfg.get("description", ""),
                hyperparameters=method_cfg.get("hyperparameters", {}),
                grid_search=method_cfg.get("grid_search", False),
                tasks=method_cfg.get("tasks"),
                layer_mapping=method_cfg.get("layer_mapping"),
                memory_optimization=method_cfg.get("memory_optimization")
            )
        
        # Training
        train = self._raw_config.get("training", {})
        opt = train.get("optimizer", {})
        batch = train.get("batch", {})
        self.training = TrainingConfig(
            learning_rate=opt.get("learning_rate", 2e-5),
            weight_decay=opt.get("weight_decay", 0.01),
            warmup_ratio=train.get("scheduler", {}).get("warmup_ratio", 0.1),
            scheduler=train.get("scheduler", {}).get("name", "cosine"),
            per_device_train_batch=batch.get("per_device_train", {"fast": 2, "full": 4}),
            per_device_eval_batch=batch.get("per_device_eval", {"fast": 4, "full": 8}),
            gradient_accumulation=batch.get("gradient_accumulation_steps", {"fast": 4, "full": 4}),
            epochs=train.get("epochs", {"fast": 1, "full": 3}),
            seeds=train.get("seeds", {"fast": [42], "full": [42, 123, 456]})
        )
        
        # Benchmarking
        bench = self._raw_config.get("benchmarking", {})
        lat = bench.get("latency", {})
        thr = bench.get("throughput", {})
        self.benchmark = BenchmarkConfig(
            warmup_runs=lat.get("warmup_runs", 3),
            measurement_runs=lat.get("measurement_runs", 10),
            batch_size=lat.get("batch_size", 1),
            throughput_batch_sizes=thr.get("batch_sizes", [1, 4, 8]),
            num_samples=thr.get("num_samples", 100),
            cpu_benchmark_enabled=bench.get("cpu_benchmark", {}).get("enabled", True),
            constrained_max_threads=bench.get("cpu_benchmark", {}).get("constrained", {}).get("max_threads", 4)
        )
        
        # Output
        out = self._raw_config.get("output", {})
        dirs = out.get("dirs", {})
        fig_settings = out.get("figure_settings", {})
        self.output = OutputConfig(
            results_dir=dirs.get("results", "./results"),
            raw_runs_dir=dirs.get("raw_runs", "./results/raw_runs"),
            summary_dir=dirs.get("summary", "./results/summary"),
            figures_dir=dirs.get("figures", "./results/figures"),
            models_dir=dirs.get("models", "./results/models"),
            figure_dpi=fig_settings.get("dpi", 300),
            figure_format=fig_settings.get("format", "png")
        )
        
        # Caching
        cache = self._raw_config.get("caching", {})
        self.cache_dir = cache.get("teacher_cache", {}).get("cache_dir", "./results/teacher_cache")
        self.cache_logits = cache.get("teacher_cache", {}).get("cache_logits", True)
        self.cache_answers = cache.get("teacher_cache", {}).get("cache_answers", True)
        self.cache_hiddens = cache.get("teacher_cache", {}).get("cache_hiddens", True)
        
        # Quantization
        quant = self._raw_config.get("quantization", {})
        self.quantization_enabled = quant.get("enabled", True)
        self.quantization_top_k = quant.get("apply_to_top_k", 2)
        self.quantization_methods = quant.get("methods", [])
        self.quantization_fallback = quant.get("fallback_on_error", True)
    
    @property
    def mode(self) -> str:
        """Return current mode string."""
        return "fast" if self.fast_mode else "full"
    
    def get_batch_size(self, train: bool = True) -> int:
        """Get batch size for current mode."""
        if train:
            return self.training.per_device_train_batch[self.mode]
        return self.training.per_device_eval_batch[self.mode]
    
    def get_grad_accum(self) -> int:
        """Get gradient accumulation steps for current mode."""
        return self.training.gradient_accumulation[self.mode]
    
    def get_epochs(self) -> int:
        """Get number of epochs for current mode."""
        return self.training.epochs[self.mode]
    
    def get_seeds(self) -> List[int]:
        """Get seeds for current mode."""
        return self.training.seeds[self.mode]
    
    def get_max_length(self, task: str) -> int:
        """Get max sequence length for task and mode."""
        return self.tasks[task].max_length[self.mode]
    
    def get_subset_size(self, task: str, split: str) -> Optional[int]:
        """Get dataset subset size for task, split, and mode."""
        return self.tasks[task].subset_sizes[self.mode].get(split)
    
    def get_device(self) -> str:
        """Get the best available device."""
        import torch
        for device in self.device.preference:
            if device == "mps" and torch.backends.mps.is_available():
                return "mps"
            elif device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "cpu":
                return "cpu"
        return "cpu"
    
    def get_precision(self) -> str:
        """Get precision for current device."""
        device = self.get_device()
        return self.device.precision.get(device, "fp32")
    
    def get_torch_dtype(self):
        """Get torch dtype based on precision setting."""
        import torch
        precision = self.get_precision()
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        return torch.float32
    
    def ensure_dirs(self):
        """Create all output directories."""
        for dir_path in [
            self.output.results_dir,
            self.output.raw_runs_dir,
            self.output.summary_dir,
            self.output.figures_dir,
            self.output.models_dir,
            self.cache_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "experiment_name": self.experiment_name,
            "fast_mode": self.fast_mode,
            "device": self.get_device(),
            "precision": self.get_precision(),
            "teacher_primary": self.teacher.primary,
            "teacher_fallback": self.teacher.local_fallback,
            "student_s1": self.student_s1.name,
            "student_s2": self.student_s2.name,
            "epochs": self.get_epochs(),
            "batch_size": self.get_batch_size(),
            "seeds": self.get_seeds(),
        }


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """
    Load experiment configuration.
    
    Args:
        config_path: Optional path to config file.
        
    Returns:
        ExperimentConfig instance
    """
    return ExperimentConfig(config_path)


# Convenience function for notebooks
def get_config() -> ExperimentConfig:
    """Get configuration with automatic path detection."""
    return load_config()
