"""
SQuAD v1.1 data loading and preprocessing utilities.

This module handles loading the SQuAD v1.1 extractive QA dataset
and preprocessing for language models.

Corresponds to thesis Section 3.6 - Tasks and Datasets.
"""

import re
import string
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter

from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer


@dataclass
class SQuADExample:
    """A single SQuAD example."""
    id: str
    question: str
    context: str
    answers: List[str]
    answer_start: List[int]


def load_squad_dataset(
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load SQuAD v1.1 dataset.
    
    Note: SQuAD test set is hidden, so we use validation as test.
    
    Args:
        subset_train: If set, sample this many training examples
        subset_val: If set, sample this many validation examples
        seed: Random seed for subsetting
        cache_dir: HuggingFace cache directory
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    dataset = load_dataset("squad", cache_dir=cache_dir)
    
    # Subset if requested
    if subset_train is not None:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(
            range(min(subset_train, len(dataset["train"])))
        )
    
    if subset_val is not None:
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(
            range(min(subset_val, len(dataset["validation"])))
        )
    
    return dataset


def create_squad_prompt(
    question: str,
    context: str,
    include_answer: bool = False,
    answer: Optional[str] = None
) -> str:
    """
    Create a prompt for SQuAD extractive QA.
    
    Args:
        question: The question to answer
        context: The context paragraph
        include_answer: Whether to include the answer (for training)
        answer: The answer text
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Answer the question based on the context below. Give a short, exact answer from the context.

Context: {context}

Question: {question}

Answer:"""
    
    if include_answer and answer is not None:
        prompt = f"{prompt} {answer}"
    
    return prompt


def tokenize_squad_for_lm(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    include_labels: bool = True
) -> Dict[str, List]:
    """
    Tokenize SQuAD examples for causal language model training.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        include_labels: Whether to include labels (for training)
        
    Returns:
        Tokenized examples
    """
    prompts = []
    answers = []
    
    for question, context, answer_dict in zip(
        examples["question"],
        examples["context"],
        examples["answers"]
    ):
        # Get first answer
        answer = answer_dict["text"][0] if answer_dict["text"] else ""
        answers.append(answer)
        
        prompt = create_squad_prompt(
            question, context,
            include_labels, answer if include_labels else None
        )
        prompts.append(prompt)
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    
    if include_labels:
        # Create labels (mask prompt, only predict answer)
        tokenized["labels"] = []
        
        for i, (ids, answer) in enumerate(zip(tokenized["input_ids"], answers)):
            # Encode answer to find its tokens
            answer_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
            
            # Find answer position in sequence
            answer_start = None
            for j in range(len(ids) - len(answer_ids) + 1):
                if ids[j:j+len(answer_ids)] == answer_ids:
                    answer_start = j
                    break
            
            # Create labels: -100 for non-answer tokens
            labels = [-100] * len(ids)
            if answer_start is not None:
                for j in range(answer_start, min(answer_start + len(answer_ids), len(ids))):
                    labels[j] = ids[j]
            
            tokenized["labels"].append(labels)
    
    # Store metadata for evaluation
    tokenized["question"] = examples["question"]
    tokenized["context"] = examples["context"]
    tokenized["gold_answers"] = [a["text"] for a in examples["answers"]]
    tokenized["example_id"] = examples["id"]
    
    return tokenized


def prepare_squad_dataset(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load and prepare SQuAD dataset for training.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        subset_train: Number of training examples (None for all)
        subset_val: Number of validation examples (None for all)
        seed: Random seed
        cache_dir: Cache directory
        
    Returns:
        Tokenized DatasetDict ready for training
    """
    # Load raw dataset
    dataset = load_squad_dataset(subset_train, subset_val, seed, cache_dir)
    
    # Define tokenization function
    def tokenize_fn(examples):
        return tokenize_squad_for_lm(examples, tokenizer, max_length)
    
    # Get columns to remove (but keep what we need)
    cols_to_remove = ["title", "context", "question", "answers"]
    cols_to_remove = [c for c in cols_to_remove if c in dataset["train"].column_names]
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenizing SQuAD"
    )
    
    return tokenized_dataset


# ============================================================================
# Evaluation utilities (EM and F1 following official SQuAD script)
# ============================================================================

def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    
    Following official SQuAD evaluation script normalization.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def compute_squad_metrics(
    predictions: List[str],
    references: List[List[str]]
) -> Dict[str, float]:
    """
    Compute SQuAD evaluation metrics.
    
    Args:
        predictions: List of predicted answer strings
        references: List of lists of gold answer strings (multiple valid answers)
        
    Returns:
        Dictionary with exact_match and f1 scores
    """
    total_em = 0.0
    total_f1 = 0.0
    
    for pred, golds in zip(predictions, references):
        # Take max over all gold answers
        em_scores = [compute_exact_match(pred, gold) for gold in golds]
        f1_scores = [compute_f1(pred, gold) for gold in golds]
        
        total_em += max(em_scores) if em_scores else 0.0
        total_f1 += max(f1_scores) if f1_scores else 0.0
    
    n = len(predictions)
    
    return {
        "exact_match": 100.0 * total_em / n if n > 0 else 0.0,
        "f1": 100.0 * total_f1 / n if n > 0 else 0.0
    }


def extract_answer_from_generation(
    generated_text: str,
    prompt: str,
    max_answer_length: int = 50
) -> str:
    """
    Extract answer from generated text.
    
    Args:
        generated_text: Full generated text including prompt
        prompt: Original prompt
        max_answer_length: Maximum answer length in characters
        
    Returns:
        Extracted answer string
    """
    # Get only the generated part
    if prompt in generated_text:
        answer = generated_text[len(prompt):].strip()
    else:
        # Try to find "Answer:" and get text after
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()
    
    # Clean up: take first line/sentence
    answer = answer.split("\n")[0].strip()
    
    # Truncate if too long
    if len(answer) > max_answer_length:
        # Try to find a natural break point
        for sep in [". ", ", ", " "]:
            if sep in answer[:max_answer_length]:
                idx = answer[:max_answer_length].rfind(sep)
                answer = answer[:idx]
                break
        else:
            answer = answer[:max_answer_length]
    
    return answer.strip()


def prepare_squad_for_qa_model(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 30
) -> Dict[str, List]:
    """
    Prepare SQuAD examples for extractive QA model (AutoModelForQuestionAnswering).
    
    This handles the span-based approach where the model predicts start/end positions.
    
    Args:
        examples: Batch of examples
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        doc_stride: Stride for splitting long documents
        max_answer_length: Maximum answer length in tokens
        
    Returns:
        Tokenized examples with start/end positions
    """
    # Tokenize questions and contexts
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None
    )
    
    # Map from tokenized examples back to original examples
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []
    tokenized["example_id"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        
        # Store example ID
        tokenized["example_id"].append(examples["id"][sample_idx])
        
        if len(answers["answer_start"]) == 0:
            # No answer (shouldn't happen in SQuAD v1.1)
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
            continue
        
        # Get answer position
        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end = answer_start + len(answer_text)
        
        # Find token positions for the context
        sequence_ids = tokenized.sequence_ids(i)
        
        # Find start and end of context in token sequence
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1
        
        # Check if answer is within this chunk
        if (offsets[context_start][0] > answer_end or 
            offsets[context_end][1] < answer_start):
            # Answer not in this chunk
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            # Find token start position
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start:
                token_start += 1
            token_start -= 1
            
            # Find token end position
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end:
                token_end -= 1
            token_end += 1
            
            tokenized["start_positions"].append(token_start)
            tokenized["end_positions"].append(token_end)
    
    return tokenized
