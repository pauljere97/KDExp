"""
Custom trainers for knowledge distillation.

This module provides custom HuggingFace Trainer classes that support
MPS device and various KD training modes.

Corresponds to thesis Section 3.8 - Training Setup.
"""

import os
import math
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    EvalPrediction,
    TrainerCallback
)
from transformers.trainer_utils import EvalLoopOutput
from tqdm import tqdm

from .kd_losses import SoftTargetLoss, SequenceKDLoss, FeatureMatchingLoss, CombinedKDLoss

logger = logging.getLogger(__name__)


class MPSTrainingArguments(TrainingArguments):
    """
    Training arguments with MPS-specific defaults.
    
    Sets sensible defaults for Apple Silicon MPS training.
    """
    
    def __init__(
        self,
        *args,
        use_mps_device: bool = True,
        **kwargs
    ):
        # Set MPS-friendly defaults
        kwargs.setdefault("dataloader_pin_memory", False)  # Avoid MPS issues
        kwargs.setdefault("fp16", False)  # MPS fp16 can be unstable
        kwargs.setdefault("bf16", False)  # MPS doesn't support bf16
        kwargs.setdefault("dataloader_num_workers", 0)  # Avoid multiprocessing issues
        
        super().__init__(*args, **kwargs)
        self.use_mps_device = use_mps_device


class BaseKDTrainer(Trainer):
    """
    Base trainer for knowledge distillation with MPS support.
    
    Provides common functionality for KD training including:
    - Teacher model management
    - KD loss computation
    - MPS memory management
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        teacher_model: Optional[PreTrainedModel] = None,
        kd_loss: Optional[nn.Module] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )
        
        self.teacher_model = teacher_model
        self.kd_loss = kd_loss
        
        # Freeze teacher if provided
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def _move_teacher_to_device(self):
        """Move teacher to the same device as student."""
        if self.teacher_model is not None:
            device = next(self.model.parameters()).device
            self.teacher_model.to(device)
    
    def _maybe_clear_cache(self, step: int, clear_every: int = 10):
        """Clear device cache periodically on MPS."""
        if step % clear_every == 0:
            device = next(self.model.parameters()).device
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


class LogitKDTrainer(BaseKDTrainer):
    """
    Trainer for logit-based knowledge distillation (KD1).
    
    Uses soft targets from teacher model with temperature scaling.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        teacher_model: PreTrainedModel,
        temperature: float = 4.0,
        alpha: float = 0.5,
        cached_teacher_logits: Optional[torch.Tensor] = None,
        **kwargs
    ):
        kd_loss = SoftTargetLoss(temperature=temperature, alpha=alpha)
        super().__init__(
            model=model,
            args=args,
            teacher_model=teacher_model,
            kd_loss=kd_loss,
            **kwargs
        )
        self.cached_teacher_logits = cached_teacher_logits
        self.current_batch_idx = 0
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute KD1 loss."""
        # Get student outputs
        labels = inputs.pop("labels", None)
        student_outputs = model(**inputs, output_hidden_states=False)
        student_logits = student_outputs.logits
        
        # Get teacher logits
        if self.cached_teacher_logits is not None:
            # Use cached logits
            batch_size = student_logits.size(0)
            start_idx = self.current_batch_idx * batch_size
            end_idx = start_idx + batch_size
            teacher_logits = self.cached_teacher_logits[start_idx:end_idx].to(student_logits.device)
            self.current_batch_idx += 1
        else:
            # Compute on-the-fly
            self._move_teacher_to_device()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
        
        # Compute KD loss
        loss = self.kd_loss(student_logits, teacher_logits, labels)
        
        return (loss, student_outputs) if return_outputs else loss


class SequenceKDTrainer(BaseKDTrainer):
    """
    Trainer for sequence-level knowledge distillation (KD2).
    
    Trains student on teacher-generated sequences.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        teacher_answers: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        **kwargs
    ):
        kd_loss = SequenceKDLoss()
        super().__init__(
            model=model,
            args=args,
            kd_loss=kd_loss,
            tokenizer=tokenizer,
            **kwargs
        )
        self.teacher_answers = teacher_answers
        self.max_length = max_length
        self._prepare_teacher_sequences()
    
    def _prepare_teacher_sequences(self):
        """Tokenize teacher-generated answers."""
        self.teacher_token_ids = {}
        
        for item in self.teacher_answers:
            example_id = item["example_id"]
            answer = item["teacher_answer"]
            prompt = item.get("prompt", "")
            
            # Create full sequence with teacher answer
            full_text = prompt + " " + answer if prompt else answer
            tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.teacher_token_ids[example_id] = tokens["input_ids"].squeeze(0)
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute KD2 sequence loss."""
        # Get example IDs
        example_ids = inputs.pop("example_id", None)
        inputs.pop("labels", None)  # Don't use original labels
        
        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Get teacher token IDs for this batch
        if example_ids is not None:
            teacher_ids = torch.stack([
                self.teacher_token_ids[eid.item() if torch.is_tensor(eid) else eid]
                for eid in example_ids
            ]).to(student_logits.device)
        else:
            # Fallback to using input_ids as pseudo-targets (not ideal)
            teacher_ids = inputs["input_ids"]
        
        # Compute sequence loss
        loss = self.kd_loss(student_logits, teacher_ids)
        
        return (loss, student_outputs) if return_outputs else loss


class FeatureKDTrainer(BaseKDTrainer):
    """
    Trainer for feature-based knowledge distillation (KD3).
    
    Matches intermediate hidden states between teacher and student.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        teacher_model: PreTrainedModel,
        layer_mapping: Dict[int, int],
        lambda_feature: float = 0.5,
        cached_teacher_hiddens: Optional[Dict[int, torch.Tensor]] = None,
        include_logit_loss: bool = True,
        temperature: float = 4.0,
        alpha: float = 0.5,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            teacher_model=teacher_model,
            **kwargs
        )
        
        self.layer_mapping = layer_mapping
        self.teacher_layers = list(layer_mapping.keys())
        self.student_layers = list(layer_mapping.values())
        
        # Feature matching loss
        self.feature_loss = FeatureMatchingLoss(
            layer_mapping=layer_mapping,
            lambda_feature=lambda_feature
        )
        
        # Optionally include logit loss
        self.include_logit_loss = include_logit_loss
        if include_logit_loss:
            self.logit_loss = SoftTargetLoss(temperature=temperature, alpha=alpha)
        
        self.cached_teacher_hiddens = cached_teacher_hiddens
        self.current_batch_idx = 0
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute KD3 feature matching loss."""
        labels = inputs.pop("labels", None)
        attention_mask = inputs.get("attention_mask")
        
        # Get student outputs with hidden states
        student_outputs = model(**inputs, output_hidden_states=True)
        student_logits = student_outputs.logits
        student_hiddens = {
            layer: student_outputs.hidden_states[layer]
            for layer in self.student_layers
        }
        
        # Get teacher outputs
        if self.cached_teacher_hiddens is not None:
            # Load cached hidden states
            batch_size = student_logits.size(0)
            start_idx = self.current_batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            teacher_hiddens = {}
            for layer in self.teacher_layers:
                cached = self.cached_teacher_hiddens[layer][start_idx:end_idx]
                teacher_hiddens[layer] = cached.to(student_logits.device)
            
            teacher_logits = None  # No cached logits
            self.current_batch_idx += 1
        else:
            # Compute on-the-fly
            self._move_teacher_to_device()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    **inputs, output_hidden_states=True
                )
                teacher_logits = teacher_outputs.logits
                teacher_hiddens = {
                    layer: teacher_outputs.hidden_states[layer]
                    for layer in self.teacher_layers
                }
        
        # Compute feature loss
        loss = self.feature_loss(teacher_hiddens, student_hiddens, attention_mask)
        
        # Add logit loss if enabled
        if self.include_logit_loss and teacher_logits is not None:
            logit_loss = self.logit_loss(student_logits, teacher_logits, labels)
            loss = loss + logit_loss
        
        return (loss, student_outputs) if return_outputs else loss


class BaselineTrainer(Trainer):
    """
    Standard trainer for baseline (B0) fine-tuning without KD.
    
    Includes MPS-specific optimizations.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        clear_cache_steps: int = 10,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.clear_cache_steps = clear_cache_steps
        self._step_count = 0
    
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Training step with cache clearing."""
        loss = super().training_step(model, inputs, **kwargs)
        
        self._step_count += 1
        if self._step_count % self.clear_cache_steps == 0:
            device = next(model.parameters()).device
            if device.type == "mps":
                torch.mps.empty_cache()
        
        return loss


def create_training_arguments(
    output_dir: str,
    num_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    save_strategy: str = "epoch",
    evaluation_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    logging_steps: int = 50,
    seed: int = 42,
    device: str = "mps",
    fp16: bool = False,
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments with sensible defaults for MPS.
    
    Args:
        output_dir: Directory for saving checkpoints
        num_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay
        save_strategy: Save strategy
        evaluation_strategy: Evaluation strategy
        load_best_model_at_end: Load best model at end
        metric_for_best_model: Metric for best model selection
        greater_is_better: Whether higher metric is better
        logging_steps: Logging frequency
        seed: Random seed
        device: Device type (mps, cuda, cpu)
        fp16: Use mixed precision (not recommended for MPS)
        
    Returns:
        TrainingArguments instance
    """
    # Disable features that don't work well on MPS
    use_mps = device == "mps"
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        save_strategy=save_strategy,
        eval_strategy=evaluation_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=logging_steps,
        seed=seed,
        # MPS-specific settings
        fp16=fp16 and not use_mps,  # Disable on MPS
        bf16=False,  # Not supported on MPS
        dataloader_pin_memory=not use_mps,
        dataloader_num_workers=0 if use_mps else 4,
        # General settings
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard by default
        **kwargs
    )


def train_with_early_stopping(
    trainer: Trainer,
    patience: int = 3,
    min_delta: float = 0.001
) -> Dict[str, Any]:
    """
    Train with early stopping.
    
    Args:
        trainer: Trainer instance
        patience: Number of evaluations to wait for improvement
        min_delta: Minimum improvement to reset patience
        
    Returns:
        Training results
    """
    from transformers import EarlyStoppingCallback
    
    callback = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=min_delta
    )
    trainer.add_callback(callback)
    
    return trainer.train()
