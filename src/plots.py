"""
Plotting utilities for thesis figures.

This module generates all figures required for Chapter 4:
- Performance vs model size
- Latency vs model size
- Throughput comparison
- Memory usage
- Pareto frontier (quality vs latency)
- KD gain vs student size

Uses matplotlib only (no seaborn dependency).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# Default plot style
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False
}

# Color palette
COLORS = {
    "teacher": "#2E86AB",
    "student_s1": "#A23B72",
    "student_s2": "#F18F01",
    "baseline": "#C73E1D",
    "kd1": "#3A7D44",
    "kd2": "#7768AE",
    "kd3": "#E76F51",
    "quantized": "#264653"
}

# Marker styles
MARKERS = {
    "teacher": "s",
    "student_s1": "o",
    "student_s2": "^",
    "baseline": "X",
    "kd1": "D",
    "kd2": "p",
    "kd3": "h"
}


def setup_plot_style():
    """Set up matplotlib style."""
    plt.rcParams.update(PLOT_STYLE)


def plot_performance_vs_model_size(
    data: pd.DataFrame,
    metric: str = "accuracy",
    task: str = "SST-2",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot performance vs model size (Fig 1 - main efficiency plot).
    
    Args:
        data: DataFrame with columns: model_name, variant, model_size_gb, {metric}
        metric: Performance metric to plot
        task: Task name for title
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by variant type
    for variant in data["variant"].unique():
        subset = data[data["variant"] == variant]
        color = COLORS.get(variant.split("_")[0], "#666666")
        marker = MARKERS.get(variant.split("_")[0], "o")
        
        ax.scatter(
            subset["model_size_gb"],
            subset[metric],
            c=color,
            marker=marker,
            s=100,
            label=variant,
            alpha=0.8,
            edgecolors="white",
            linewidths=1
        )
    
    ax.set_xlabel("Model Size (GB)")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
    ax.set_title(f"{task}: {metric.replace('_', ' ').title()} vs Model Size")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    return fig


def plot_latency_vs_model_size(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot latency vs model size (Fig 2).
    
    Args:
        data: DataFrame with columns: model_name, variant, model_size_gb, latency_ms_per_token
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each variant
    for variant in data["variant"].unique():
        subset = data[data["variant"] == variant]
        color = COLORS.get(variant.split("_")[0], "#666666")
        marker = MARKERS.get(variant.split("_")[0], "o")
        
        ax.scatter(
            subset["model_size_gb"],
            subset["latency_ms_per_token"],
            c=color,
            marker=marker,
            s=100,
            label=variant,
            alpha=0.8,
            edgecolors="white",
            linewidths=1
        )
    
    # Add trend line
    x = data["model_size_gb"].values
    y = data["latency_ms_per_token"].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, label="Trend")
    
    ax.set_xlabel("Model Size (GB)")
    ax.set_ylabel("Latency (ms/token)")
    ax.set_title("Inference Latency vs Model Size")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    return fig


def plot_throughput_comparison(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot throughput comparison bar chart (Fig 3).
    
    Args:
        data: DataFrame with columns: model_name, variant, throughput_tokens_per_sec
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for bar plot
    variants = data["variant"].unique()
    x = np.arange(len(variants))
    width = 0.6
    
    throughputs = [data[data["variant"] == v]["throughput_tokens_per_sec"].mean() for v in variants]
    stds = [data[data["variant"] == v]["throughput_tokens_per_sec"].std() for v in variants]
    
    colors = [COLORS.get(v.split("_")[0], "#666666") for v in variants]
    
    bars = ax.bar(x, throughputs, width, yerr=stds, color=colors, alpha=0.8,
                  edgecolor="white", linewidth=1, capsize=3)
    
    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Inference Throughput Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    
    # Add value labels on bars
    for bar, val in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    return fig


def plot_memory_usage(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot memory usage comparison (Fig 4).
    
    Args:
        data: DataFrame with columns: model_name, variant, peak_ram_gb, peak_device_memory_gb
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    variants = data["variant"].unique()
    x = np.arange(len(variants))
    width = 0.35
    
    ram = [data[data["variant"] == v]["peak_ram_gb"].mean() for v in variants]
    device_mem = [data[data["variant"] == v]["peak_device_memory_gb"].mean() for v in variants]
    
    bars1 = ax.bar(x - width/2, ram, width, label="Peak RAM", color=COLORS["teacher"], alpha=0.8)
    bars2 = ax.bar(x + width/2, device_mem, width, label="Device Memory", color=COLORS["student_s1"], alpha=0.8)
    
    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Memory Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    return fig


def plot_pareto_quality_vs_latency(
    data: pd.DataFrame,
    metric: str = "accuracy",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot Pareto frontier of quality vs latency (Fig 5).
    
    Args:
        data: DataFrame with columns: variant, {metric}, latency_ms_per_token
        metric: Quality metric
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    for variant in data["variant"].unique():
        subset = data[data["variant"] == variant]
        color = COLORS.get(variant.split("_")[0], "#666666")
        marker = MARKERS.get(variant.split("_")[0], "o")
        
        ax.scatter(
            subset["latency_ms_per_token"],
            subset[metric],
            c=color,
            marker=marker,
            s=100,
            label=variant,
            alpha=0.8,
            edgecolors="white",
            linewidths=1
        )
    
    # Find and plot Pareto frontier
    latencies = data["latency_ms_per_token"].values
    qualities = data[metric].values
    
    pareto_mask = np.ones(len(data), dtype=bool)
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                # Point j dominates point i if it has lower latency AND higher quality
                if latencies[j] <= latencies[i] and qualities[j] >= qualities[i]:
                    if latencies[j] < latencies[i] or qualities[j] > qualities[i]:
                        pareto_mask[i] = False
                        break
    
    pareto_points = data[pareto_mask].sort_values("latency_ms_per_token")
    if len(pareto_points) > 1:
        ax.plot(
            pareto_points["latency_ms_per_token"],
            pareto_points[metric],
            "k--",
            alpha=0.5,
            linewidth=2,
            label="Pareto Frontier"
        )
    
    ax.set_xlabel("Latency (ms/token)")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
    ax.set_title("Pareto Frontier: Quality vs Latency")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    return fig


def plot_kd_gain_vs_student_size(
    data: pd.DataFrame,
    baseline_metric: str = "accuracy",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot KD performance gain vs student model size (Fig 6).
    
    Args:
        data: DataFrame with columns: variant, params_b, {baseline_metric}, delta_vs_b0
        baseline_metric: Metric to show
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to KD variants only
    kd_variants = [v for v in data["variant"].unique() if v.startswith(("kd1", "kd2", "kd3", "KD"))]
    
    for variant in kd_variants:
        subset = data[data["variant"] == variant]
        color = COLORS.get(variant.split("_")[0].lower(), "#666666")
        marker = MARKERS.get(variant.split("_")[0].lower(), "o")
        
        ax.scatter(
            subset["params_b"],
            subset["delta_vs_b0"],
            c=color,
            marker=marker,
            s=100,
            label=variant,
            alpha=0.8,
            edgecolors="white",
            linewidths=1
        )
    
    # Add zero line
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Model Parameters (B)")
    ax.set_ylabel("Performance Gain vs Baseline")
    ax.set_title("Knowledge Distillation Gain vs Model Size")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    return fig


def plot_ablation_heatmap(
    data: pd.DataFrame,
    x_param: str = "temperature",
    y_param: str = "alpha",
    metric: str = "accuracy",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ablation study heatmap.
    
    Args:
        data: DataFrame with ablation results
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        metric: Metric to visualize
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot data for heatmap
    pivot = data.pivot_table(
        values=metric,
        index=y_param,
        columns=x_param,
        aggfunc="mean"
    )
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.replace("_", " ").title(), rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            text = ax.text(j, i, f"{value:.3f}",
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel(x_param.replace("_", " ").title())
    ax.set_ylabel(y_param.replace("_", " ").title())
    ax.set_title(f"Ablation Study: {metric.replace('_', ' ').title()}")
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")
    
    return fig


def generate_all_figures(
    results_df: pd.DataFrame,
    output_dir: str,
    task: str = "SST-2",
    metric: str = "accuracy"
):
    """
    Generate all thesis figures.
    
    Args:
        results_df: Combined results DataFrame
        output_dir: Directory to save figures
        task: Task name
        metric: Primary metric
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Fig 1: Performance vs Model Size
    plot_performance_vs_model_size(
        results_df, metric=metric, task=task,
        output_path=str(output_path / "performance_vs_model_size.png")
    )
    
    # Fig 2: Latency vs Model Size
    plot_latency_vs_model_size(
        results_df,
        output_path=str(output_path / "latency_vs_model_size.png")
    )
    
    # Fig 3: Throughput Comparison
    plot_throughput_comparison(
        results_df,
        output_path=str(output_path / "throughput_comparison.png")
    )
    
    # Fig 4: Memory Usage
    plot_memory_usage(
        results_df,
        output_path=str(output_path / "memory_usage.png")
    )
    
    # Fig 5: Pareto Frontier
    plot_pareto_quality_vs_latency(
        results_df, metric=metric,
        output_path=str(output_path / "pareto_quality_vs_latency.png")
    )
    
    # Fig 6: KD Gain vs Size
    if "delta_vs_b0" in results_df.columns:
        plot_kd_gain_vs_student_size(
            results_df,
            output_path=str(output_path / "kd_gain_vs_student_size.png")
        )
    
    logger.info(f"Generated all figures in {output_dir}")
