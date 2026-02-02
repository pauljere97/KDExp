"""
SST-2 data loading and preprocessing utilities.

This module handles loading the GLUE SST-2 sentiment classification dataset
and preprocessing for language models.

Corresponds to thesis Section 3.6 - Tasks and Datasets.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer


@dataclass
class SST2Example:
    """A single SST-2 example."""
    idx: int
    sentence: str
    label: int
    label_name: str


def load_sst2_dataset(
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load SST-2 dataset from GLUE benchmark.
    
    Args:
        subset_train: If set, sample this many training examples
        subset_val: If set, sample this many validation examples
        seed: Random seed for subsetting
        cache_dir: HuggingFace cache directory
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)
    
    # Subset if requested
    if subset_train is not None:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(min(subset_train, len(dataset["train"]))))
    
    if subset_val is not None:
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(min(subset_val, len(dataset["validation"]))))
    
    return dataset


def create_sst2_prompt(sentence: str, include_label: bool = False, label: Optional[int] = None) -> str:
    """
    Create a prompt for SST-2 sentiment classification.
    
    Args:
        sentence: The sentence to classify
        include_label: Whether to include the label in the prompt (for training)
        label: The label (0=negative, 1=positive)
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Classify the sentiment of the following sentence as positive or negative.

Sentence: {sentence}

Sentiment:"""
    
    if include_label and label is not None:
        label_text = "positive" if label == 1 else "negative"
        prompt = f"{prompt} {label_text}"
    
    return prompt


def tokenize_sst2_for_lm(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    include_labels: bool = True
) -> Dict[str, List]:
    """
    Tokenize SST-2 examples for causal language model training.
    
    Creates prompt format and tokenizes with labels for next-token prediction.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        include_labels: Whether to include labels (for training)
        
    Returns:
        Tokenized examples
    """
    prompts = []
    for sentence, label in zip(examples["sentence"], examples["label"]):
        prompt = create_sst2_prompt(sentence, include_labels, label if include_labels else None)
        prompts.append(prompt)
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None  # Return lists for datasets
    )
    
    if include_labels:
        # For causal LM, labels are the same as input_ids (shifted internally)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        # Mask prompt tokens (only predict the sentiment word)
        # Find where the label starts and mask everything before
        for i, (ids, label) in enumerate(zip(tokenized["input_ids"], examples["label"])):
            label_text = "positive" if label == 1 else "negative"
            label_ids = tokenizer.encode(f" {label_text}", add_special_tokens=False)
            
            # Find the position of the label in the sequence
            label_start = None
            for j in range(len(ids) - len(label_ids) + 1):
                if ids[j:j+len(label_ids)] == label_ids:
                    label_start = j
                    break
            
            # Create labels: -100 for prompt, actual IDs for label
            labels = [-100] * len(ids)
            if label_start is not None:
                for j in range(label_start, min(label_start + len(label_ids), len(ids))):
                    labels[j] = ids[j]
            
            tokenized["labels"][i] = labels
    
    # Store original labels for evaluation
    tokenized["original_labels"] = examples["label"]
    
    return tokenized


def tokenize_sst2_for_classification(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256
) -> Dict[str, List]:
    """
    Tokenize SST-2 for sequence classification head.
    
    This is an alternative approach using a classification head
    instead of generative classification.
    
    Args:
        examples: Batch of examples
        tokenizer: HuggingFace tokenizer  
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples with labels
    """
    tokenized = tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    
    tokenized["labels"] = examples["label"]
    
    return tokenized


def prepare_sst2_dataset(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    seed: int = 42,
    use_classification_head: bool = False,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load and prepare SST-2 dataset for training.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        subset_train: Number of training examples (None for all)
        subset_val: Number of validation examples (None for all)
        seed: Random seed
        use_classification_head: If True, prepare for classification head
        cache_dir: Cache directory
        
    Returns:
        Tokenized DatasetDict ready for training
    """
    # Load raw dataset
    dataset = load_sst2_dataset(subset_train, subset_val, seed, cache_dir)
    
    # Choose tokenization function
    if use_classification_head:
        tokenize_fn = lambda x: tokenize_sst2_for_classification(x, tokenizer, max_length)
    else:
        tokenize_fn = lambda x: tokenize_sst2_for_lm(x, tokenizer, max_length)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing SST-2"
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch")
    
    return tokenized_dataset


def extract_sentiment_prediction(
    generated_text: str,
    prompt: str
) -> int:
    """
    Extract sentiment prediction from generated text.
    
    Args:
        generated_text: Full generated text including prompt
        prompt: Original prompt
        
    Returns:
        Predicted label (0=negative, 1=positive, -1=unknown)
    """
    # Get only the generated part
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip().lower()
    else:
        response = generated_text.strip().lower()
    
    # Check for sentiment keywords
    response_start = response.split()[0] if response.split() else ""
    
    if "positive" in response_start or response_start.startswith("pos"):
        return 1
    elif "negative" in response_start or response_start.startswith("neg"):
        return 0
    
    # Fallback: check full response
    if "positive" in response:
        return 1
    elif "negative" in response:
        return 0
    
    return -1  # Unknown


def compute_sst2_metrics(
    predictions: List[int],
    references: List[int]
) -> Dict[str, float]:
    """
    Compute SST-2 evaluation metrics.
    
    Args:
        predictions: Predicted labels
        references: True labels
        
    Returns:
        Dictionary with accuracy and F1 scores
    """
    import evaluate
    
    # Filter out invalid predictions
    valid_preds = []
    valid_refs = []
    for pred, ref in zip(predictions, references):
        if pred != -1:
            valid_preds.append(pred)
            valid_refs.append(ref)
    
    if not valid_preds:
        return {"accuracy": 0.0, "f1": 0.0, "valid_ratio": 0.0}
    
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    accuracy = accuracy_metric.compute(predictions=valid_preds, references=valid_refs)
    f1 = f1_metric.compute(predictions=valid_preds, references=valid_refs, average="binary")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "valid_ratio": len(valid_preds) / len(predictions)
    }


def get_sst2_label_tokens(tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
    """
    Get token IDs for SST-2 label words.
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dictionary mapping label names to token IDs
    """
    # Get first token of each label word (with leading space)
    pos_ids = tokenizer.encode(" positive", add_special_tokens=False)
    neg_ids = tokenizer.encode(" negative", add_special_tokens=False)
    
    return {
        "positive": pos_ids[0] if pos_ids else tokenizer.encode("positive", add_special_tokens=False)[0],
        "negative": neg_ids[0] if neg_ids else tokenizer.encode("negative", add_special_tokens=False)[0]
    }
