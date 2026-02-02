"""
Teacher model caching utilities.

This module handles caching teacher model outputs for knowledge distillation:
- KD1: Logits (soft labels)
- KD2: Generated answer sequences (for QA)
- KD3: Intermediate hidden states

Corresponds to thesis Section 3.7 - Knowledge Distillation Methods.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)

logger = logging.getLogger(__name__)


@dataclass
class TeacherCacheConfig:
    """Configuration for teacher caching."""
    cache_dir: str
    cache_logits: bool = True
    cache_answers: bool = True
    cache_hiddens: bool = True
    hidden_layers: Optional[List[int]] = None  # Layers to cache (None = all)
    hidden_dtype: str = "float16"
    chunk_size: int = 500
    compute_on_the_fly: bool = False  # For memory-constrained setups


class TeacherCache:
    """
    Manager for caching and loading teacher model outputs.
    
    Handles efficient storage and retrieval of:
    - Logits for soft-label KD
    - Generated sequences for sequence-level KD
    - Hidden states for feature-based KD
    """
    
    def __init__(self, config: TeacherCacheConfig):
        """
        Initialize teacher cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.logits_dir = self.cache_dir / "logits"
        self.answers_dir = self.cache_dir / "answers"
        self.hiddens_dir = self.cache_dir / "hiddens"
        self.metadata_path = self.cache_dir / "metadata.json"
        
    def _get_cache_key(self, task: str, split: str, model_name: str) -> str:
        """Generate a unique cache key."""
        # Sanitize model name for filesystem
        model_key = model_name.replace("/", "_").replace("\\", "_")
        return f"{task}_{split}_{model_key}"
    
    def _save_metadata(self, key: str, metadata: Dict[str, Any]):
        """Save metadata for a cache entry."""
        all_metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                all_metadata = json.load(f)
        
        all_metadata[key] = metadata
        
        with open(self.metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
    
    def _load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a cache entry."""
        if not self.metadata_path.exists():
            return None
        
        with open(self.metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        return all_metadata.get(key)
    
    def has_cache(
        self,
        task: str,
        split: str,
        model_name: str,
        cache_type: str = "all"
    ) -> bool:
        """
        Check if cache exists.
        
        Args:
            task: Task name (sst2, squad)
            split: Data split (train, validation)
            model_name: Teacher model name
            cache_type: Type of cache (logits, answers, hiddens, all)
            
        Returns:
            True if cache exists
        """
        key = self._get_cache_key(task, split, model_name)
        metadata = self._load_metadata(key)
        
        if metadata is None:
            return False
        
        if cache_type == "all":
            return all([
                metadata.get("logits_cached", False) if self.config.cache_logits else True,
                metadata.get("answers_cached", False) if self.config.cache_answers else True,
                metadata.get("hiddens_cached", False) if self.config.cache_hiddens else True,
            ])
        
        return metadata.get(f"{cache_type}_cached", False)
    
    # =========================================================================
    # Logits caching (for KD1)
    # =========================================================================
    
    def cache_logits(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: DataLoader,
        task: str,
        split: str,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Cache teacher logits for a dataset.
        
        Args:
            model: Teacher model
            tokenizer: Tokenizer
            dataloader: DataLoader with tokenized examples
            task: Task name
            split: Data split
            model_name: Model identifier
            device: Device to run on
            dtype: Model dtype
        """
        key = self._get_cache_key(task, split, model_name)
        save_dir = self.logits_dir / key
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        model.to(device)
        
        all_logits = []
        chunk_idx = 0
        
        logger.info(f"Caching logits for {task}/{split}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching logits")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                )
                
                # Get logits and convert to half precision for storage
                logits = outputs.logits.cpu().half()
                all_logits.append(logits)
                
                # Save in chunks to manage memory
                total_examples = sum(l.shape[0] for l in all_logits)
                if total_examples >= self.config.chunk_size:
                    chunk_logits = torch.cat(all_logits, dim=0)
                    torch.save(chunk_logits, save_dir / f"chunk_{chunk_idx}.pt")
                    all_logits = []
                    chunk_idx += 1
                    
                    # Clear MPS cache if needed
                    if device == "mps":
                        torch.mps.empty_cache()
        
        # Save remaining
        if all_logits:
            chunk_logits = torch.cat(all_logits, dim=0)
            torch.save(chunk_logits, save_dir / f"chunk_{chunk_idx}.pt")
        
        # Save metadata
        metadata = self._load_metadata(key) or {}
        metadata["logits_cached"] = True
        metadata["logits_chunks"] = chunk_idx + 1
        metadata["model_name"] = model_name
        self._save_metadata(key, metadata)
        
        logger.info(f"Cached logits in {chunk_idx + 1} chunks")
    
    def load_logits(
        self,
        task: str,
        split: str,
        model_name: str,
        indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Load cached logits.
        
        Args:
            task: Task name
            split: Data split
            model_name: Model identifier
            indices: Optional list of example indices to load
            
        Returns:
            Tensor of logits
        """
        key = self._get_cache_key(task, split, model_name)
        load_dir = self.logits_dir / key
        
        metadata = self._load_metadata(key)
        if metadata is None or not metadata.get("logits_cached"):
            raise ValueError(f"No cached logits found for {key}")
        
        num_chunks = metadata["logits_chunks"]
        
        all_logits = []
        for i in range(num_chunks):
            chunk = torch.load(load_dir / f"chunk_{i}.pt", weights_only=True)
            all_logits.append(chunk)
        
        logits = torch.cat(all_logits, dim=0)
        
        if indices is not None:
            logits = logits[indices]
        
        return logits
    
    # =========================================================================
    # Answer caching (for KD2 - sequence-level KD)
    # =========================================================================
    
    def cache_answers(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        examples: List[Dict[str, Any]],
        task: str,
        split: str,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = False
    ):
        """
        Cache teacher-generated answers for sequence-level KD.
        
        Args:
            model: Teacher model
            tokenizer: Tokenizer
            examples: List of examples with prompts
            task: Task name
            split: Data split
            model_name: Model identifier
            device: Device to run on
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
        """
        key = self._get_cache_key(task, split, model_name)
        save_path = self.answers_dir / f"{key}.json"
        self.answers_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        model.to(device)
        
        generated_answers = []
        
        logger.info(f"Generating teacher answers for {task}/{split}...")
        
        with torch.no_grad():
            for example in tqdm(examples, desc="Generating answers"):
                prompt = example.get("prompt", "")
                
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract answer (remove prompt)
                if prompt in generated_text:
                    answer = generated_text[len(prompt):].strip()
                else:
                    answer = generated_text.strip()
                
                generated_answers.append({
                    "example_id": example.get("id", len(generated_answers)),
                    "prompt": prompt,
                    "teacher_answer": answer
                })
                
                # Clear MPS cache periodically
                if device == "mps" and len(generated_answers) % 100 == 0:
                    torch.mps.empty_cache()
        
        # Save answers
        with open(save_path, 'w') as f:
            json.dump(generated_answers, f, indent=2)
        
        # Update metadata
        metadata = self._load_metadata(key) or {}
        metadata["answers_cached"] = True
        metadata["num_answers"] = len(generated_answers)
        metadata["model_name"] = model_name
        self._save_metadata(key, metadata)
        
        logger.info(f"Cached {len(generated_answers)} teacher answers")
    
    def load_answers(
        self,
        task: str,
        split: str,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Load cached teacher answers.
        
        Args:
            task: Task name
            split: Data split
            model_name: Model identifier
            
        Returns:
            List of answer dictionaries
        """
        key = self._get_cache_key(task, split, model_name)
        load_path = self.answers_dir / f"{key}.json"
        
        if not load_path.exists():
            raise ValueError(f"No cached answers found at {load_path}")
        
        with open(load_path, 'r') as f:
            return json.load(f)
    
    # =========================================================================
    # Hidden states caching (for KD3 - feature-based KD)
    # =========================================================================
    
    def cache_hiddens(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: DataLoader,
        task: str,
        split: str,
        model_name: str,
        layers: List[int],
        device: str = "cpu",
        max_samples: Optional[int] = None
    ):
        """
        Cache teacher hidden states for feature-based KD.
        
        Args:
            model: Teacher model
            tokenizer: Tokenizer
            dataloader: DataLoader with tokenized examples
            task: Task name
            split: Data split
            model_name: Model identifier
            layers: Which layers to cache (0-indexed)
            device: Device to run on
            max_samples: Max samples to cache (for memory constraints)
        """
        key = self._get_cache_key(task, split, model_name)
        save_dir = self.hiddens_dir / key
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        model.to(device)
        
        # Store hiddens per layer
        layer_hiddens = {l: [] for l in layers}
        sample_count = 0
        chunk_idx = 0
        
        logger.info(f"Caching hidden states for {task}/{split}, layers {layers}...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Caching hiddens"):
                if max_samples and sample_count >= max_samples:
                    break
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states
                
                for layer_idx in layers:
                    # Get hidden state for this layer (convert to half for storage)
                    h = hidden_states[layer_idx].cpu()
                    if self.config.hidden_dtype == "float16":
                        h = h.half()
                    layer_hiddens[layer_idx].append(h)
                
                sample_count += input_ids.shape[0]
                
                # Save chunks
                total_in_chunk = sum(h.shape[0] for h in layer_hiddens[layers[0]])
                if total_in_chunk >= self.config.chunk_size:
                    for layer_idx in layers:
                        chunk_tensor = torch.cat(layer_hiddens[layer_idx], dim=0)
                        torch.save(
                            chunk_tensor,
                            save_dir / f"layer_{layer_idx}_chunk_{chunk_idx}.pt"
                        )
                        layer_hiddens[layer_idx] = []
                    chunk_idx += 1
                    
                    if device == "mps":
                        torch.mps.empty_cache()
        
        # Save remaining
        if layer_hiddens[layers[0]]:
            for layer_idx in layers:
                if layer_hiddens[layer_idx]:
                    chunk_tensor = torch.cat(layer_hiddens[layer_idx], dim=0)
                    torch.save(
                        chunk_tensor,
                        save_dir / f"layer_{layer_idx}_chunk_{chunk_idx}.pt"
                    )
        
        # Save metadata
        metadata = self._load_metadata(key) or {}
        metadata["hiddens_cached"] = True
        metadata["hiddens_chunks"] = chunk_idx + 1
        metadata["hiddens_layers"] = layers
        metadata["num_samples"] = sample_count
        metadata["model_name"] = model_name
        self._save_metadata(key, metadata)
        
        logger.info(f"Cached hidden states for {sample_count} samples in {chunk_idx + 1} chunks")
    
    def load_hiddens(
        self,
        task: str,
        split: str,
        model_name: str,
        layer: int,
        indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Load cached hidden states for a specific layer.
        
        Args:
            task: Task name
            split: Data split
            model_name: Model identifier
            layer: Layer index to load
            indices: Optional list of example indices
            
        Returns:
            Tensor of hidden states
        """
        key = self._get_cache_key(task, split, model_name)
        load_dir = self.hiddens_dir / key
        
        metadata = self._load_metadata(key)
        if metadata is None or not metadata.get("hiddens_cached"):
            raise ValueError(f"No cached hiddens found for {key}")
        
        if layer not in metadata["hiddens_layers"]:
            raise ValueError(f"Layer {layer} not cached. Available: {metadata['hiddens_layers']}")
        
        num_chunks = metadata["hiddens_chunks"]
        
        all_hiddens = []
        for i in range(num_chunks):
            chunk_path = load_dir / f"layer_{layer}_chunk_{i}.pt"
            if chunk_path.exists():
                chunk = torch.load(chunk_path, weights_only=True)
                all_hiddens.append(chunk)
        
        hiddens = torch.cat(all_hiddens, dim=0)
        
        if indices is not None:
            hiddens = hiddens[indices]
        
        return hiddens
    
    def get_layer_mapping(
        self,
        teacher_num_layers: int,
        student_num_layers: int,
        mode: str = "proportional"
    ) -> Dict[int, int]:
        """
        Get layer mapping from teacher to student.
        
        Args:
            teacher_num_layers: Number of teacher layers
            student_num_layers: Number of student layers
            mode: Mapping mode (proportional or fixed)
            
        Returns:
            Dictionary mapping teacher layer -> student layer
        """
        if mode == "proportional":
            # Map proportionally
            mapping = {}
            ratio = student_num_layers / teacher_num_layers
            for t_layer in range(teacher_num_layers):
                s_layer = int(t_layer * ratio)
                if s_layer not in mapping.values():
                    mapping[t_layer] = s_layer
            return mapping
        else:
            # Fixed mapping (example for 12 -> 6)
            return {4: 2, 8: 4, 12: 6}
    
    def clear_cache(self, task: Optional[str] = None):
        """Clear cached data."""
        import shutil
        
        if task:
            # Clear specific task
            for subdir in [self.logits_dir, self.answers_dir, self.hiddens_dir]:
                for path in subdir.glob(f"{task}_*"):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
        else:
            # Clear all
            for subdir in [self.logits_dir, self.answers_dir, self.hiddens_dir]:
                if subdir.exists():
                    shutil.rmtree(subdir)
                    subdir.mkdir()


def compute_teacher_outputs_on_the_fly(
    teacher_model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool = True,
    hidden_layers: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute teacher outputs on-the-fly (for memory-constrained setups).
    
    This function computes teacher outputs during training without caching.
    Use when caching is not feasible due to memory constraints.
    
    Args:
        teacher_model: Teacher model (should be in eval mode)
        input_ids: Input token IDs
        attention_mask: Attention mask
        output_hidden_states: Whether to return hidden states
        hidden_layers: Which layers to return (None = all)
        
    Returns:
        Dictionary with logits and optionally hidden states
    """
    with torch.no_grad():
        outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )
    
    result = {"logits": outputs.logits}
    
    if output_hidden_states and hidden_layers:
        selected_hiddens = []
        for layer_idx in hidden_layers:
            selected_hiddens.append(outputs.hidden_states[layer_idx])
        result["hidden_states"] = selected_hiddens
    elif output_hidden_states:
        result["hidden_states"] = outputs.hidden_states
    
    return result
