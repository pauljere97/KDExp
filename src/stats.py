"""
Statistical analysis utilities.

This module provides statistical tests for comparing model performance:
- Paired t-test for comparing two methods
- Cohen's d effect size
- Significance tables for thesis

Corresponds to thesis Chapter 4 - Results and Analysis.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

import numpy as np
from scipy import stats
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    comparison: str
    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    mean_diff: float
    std_a: float
    std_b: float
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool
    alpha: float = 0.05


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    method_a: str = "A",
    method_b: str = "B",
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    Perform paired t-test between two sets of scores.
    
    Args:
        scores_a: Scores from method A (e.g., across seeds)
        scores_b: Scores from method B
        method_a: Name of method A
        method_b: Name of method B
        alpha: Significance level
        
    Returns:
        StatisticalTestResult with test statistics
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length for paired t-test")
    
    if len(scores_a) < 2:
        warnings.warn("Less than 2 samples; t-test may not be meaningful")
    
    # Compute statistics
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    mean_diff = mean_b - mean_a
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)
    
    # Paired t-test
    if len(scores_a) >= 2:
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    else:
        t_stat, p_value = 0.0, 1.0
    
    # Cohen's d for paired samples
    diff = scores_b - scores_a
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    
    # Determine significance
    significant = p_value < alpha
    
    return StatisticalTestResult(
        comparison=f"{method_a} vs {method_b}",
        method_a=method_a,
        method_b=method_b,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        std_a=std_a,
        std_b=std_b,
        t_statistic=t_stat,
        p_value=p_value,
        cohens_d=cohens_d,
        significant=significant,
        alpha=alpha
    )


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cohen's d value
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    d_abs = abs(d)
    
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Args:
        scores: Score values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    scores = np.array(scores)
    n = len(scores)
    
    # Bootstrap means
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentiles
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return lower, upper


def create_significance_table(
    results: Dict[str, List[float]],
    baseline_key: str = "baseline",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Create significance table comparing all methods to baseline.
    
    Args:
        results: Dictionary mapping method names to score lists
        baseline_key: Key for baseline method
        alpha: Significance level
        
    Returns:
        DataFrame with comparison statistics
    """
    if baseline_key not in results:
        raise ValueError(f"Baseline key '{baseline_key}' not found in results")
    
    baseline_scores = results[baseline_key]
    comparisons = []
    
    for method, scores in results.items():
        if method == baseline_key:
            continue
        
        test_result = paired_t_test(
            baseline_scores, scores,
            method_a=baseline_key,
            method_b=method,
            alpha=alpha
        )
        
        comparisons.append({
            "comparison": test_result.comparison,
            "method": method,
            "mean": test_result.mean_b,
            "std": test_result.std_b,
            "baseline_mean": test_result.mean_a,
            "mean_diff": test_result.mean_diff,
            "t_statistic": test_result.t_statistic,
            "p_value": test_result.p_value,
            "cohens_d": test_result.cohens_d,
            "effect_size": interpret_cohens_d(test_result.cohens_d),
            "significant": "Yes" if test_result.significant else "No"
        })
    
    return pd.DataFrame(comparisons)


def compare_all_pairs(
    results: Dict[str, List[float]],
    alpha: float = 0.05,
    correction: str = "bonferroni"
) -> pd.DataFrame:
    """
    Compare all pairs of methods with multiple comparison correction.
    
    Args:
        results: Dictionary mapping method names to score lists
        alpha: Significance level
        correction: Multiple comparison correction method
        
    Returns:
        DataFrame with pairwise comparison statistics
    """
    methods = list(results.keys())
    n_comparisons = len(methods) * (len(methods) - 1) // 2
    
    # Adjusted alpha for multiple comparisons
    if correction == "bonferroni":
        adjusted_alpha = alpha / n_comparisons
    else:
        adjusted_alpha = alpha
    
    comparisons = []
    
    for i, method_a in enumerate(methods):
        for method_b in methods[i+1:]:
            test_result = paired_t_test(
                results[method_a],
                results[method_b],
                method_a=method_a,
                method_b=method_b,
                alpha=adjusted_alpha
            )
            
            comparisons.append({
                "method_a": method_a,
                "method_b": method_b,
                "mean_a": test_result.mean_a,
                "mean_b": test_result.mean_b,
                "mean_diff": test_result.mean_diff,
                "p_value": test_result.p_value,
                "p_value_corrected": min(test_result.p_value * n_comparisons, 1.0) if correction == "bonferroni" else test_result.p_value,
                "cohens_d": test_result.cohens_d,
                "significant": test_result.significant,
                "correction": correction
            })
    
    return pd.DataFrame(comparisons)


def summarize_results_with_ci(
    results: Dict[str, List[float]],
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Summarize results with confidence intervals.
    
    Args:
        results: Dictionary mapping method names to score lists
        confidence: Confidence level for intervals
        
    Returns:
        DataFrame with summary statistics
    """
    summaries = []
    
    for method, scores in results.items():
        scores = np.array(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        
        if len(scores) >= 2:
            ci_low, ci_high = bootstrap_ci(scores, confidence=confidence)
        else:
            ci_low, ci_high = mean, mean
        
        summaries.append({
            "method": method,
            "mean": mean,
            "std": std,
            "min": np.min(scores),
            "max": np.max(scores),
            f"ci_{int(confidence*100)}_low": ci_low,
            f"ci_{int(confidence*100)}_high": ci_high,
            "n_runs": len(scores)
        })
    
    return pd.DataFrame(summaries)


def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def format_significance_table_for_latex(
    df: pd.DataFrame,
    caption: str = "Statistical Significance Tests",
    label: str = "tab:significance"
) -> str:
    """
    Format significance table for LaTeX.
    
    Args:
        df: DataFrame with significance results
        caption: Table caption
        label: LaTeX label
        
    Returns:
        LaTeX table string
    """
    # Select and format columns
    cols = ["comparison", "mean_diff", "p_value", "cohens_d", "significant"]
    df_formatted = df[cols].copy()
    
    # Format p-values
    df_formatted["p_value"] = df_formatted["p_value"].apply(format_p_value)
    
    # Format numbers
    df_formatted["mean_diff"] = df_formatted["mean_diff"].apply(lambda x: f"{x:.4f}")
    df_formatted["cohens_d"] = df_formatted["cohens_d"].apply(lambda x: f"{x:.3f}")
    
    # Rename columns
    df_formatted.columns = ["Comparison", "Mean Diff", "p-value", "Cohen's d", "Sig."]
    
    # Generate LaTeX
    latex = df_formatted.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df_formatted.columns) - 1)
    )
    
    # Wrap in table environment
    full_latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}
"""
    return full_latex
