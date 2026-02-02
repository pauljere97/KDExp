#!/usr/bin/env python3
"""Generate thesis analysis notebook with proper JSON formatting."""

import json
from pathlib import Path

def create_notebook():
    cells = []
    
    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.split('\n')
        })
    
    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.split('\n')
        })
    
    # Cell 1: Title
    add_md("""# Thesis Analysis: Knowledge Distillation for LLMs

**Statistical Analysis & Publication-Ready Outputs**

This notebook generates:
1. Statistical significance tests (t-tests, effect sizes)
2. Publication-quality figures (Nature/IEEE style)
3. LaTeX tables for thesis chapters
4. Summary statistics with confidence intervals""")

    # Cell 2: Imports
    add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('../results')
RUNS_DIR = RESULTS_DIR / 'runs'
FIGS_DIR = RESULTS_DIR / 'figures'
FIGS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(RUNS_DIR / 'results.csv')
df_bench = pd.read_csv(RUNS_DIR / 'benchmarks.csv')

print('Results loaded!')
print(f'Training runs: {len(df)}')
print(f'Benchmark runs: {len(df_bench)}')""")

    # Cell 3: Data Overview header
    add_md("## 1. Data Overview")

    # Cell 4: Display data
    add_code("""print('='*60)
print('TRAINING RESULTS')
print('='*60)
display(df)

print('\\n' + '='*60)
print('BENCHMARK RESULTS')
print('='*60)
display(df_bench)""")

    # Cell 5: Summary stats
    add_code("""# Summary statistics
summary = df.groupby(['method', 'task'])['eval_loss'].agg(['mean', 'std', 'count']).reset_index()
summary['se'] = summary['std'] / np.sqrt(summary['count'])
summary['ci_95'] = 1.96 * summary['se']
summary['range'] = summary.apply(lambda r: f"{r['mean']:.4f} +/- {r['ci_95']:.4f}", axis=1)

print('Summary Statistics (Mean +/- 95% CI)')
print('='*60)
display(summary[['method', 'task', 'mean', 'std', 'range']])""")

    # Cell 6: Stats header
    add_md("## 2. Statistical Significance Tests")

    # Cell 7: Helper functions
    add_code("""def cohens_d(group1, group2):
    \"\"\"Calculate Cohen's d effect size\"\"\"
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

def interpret_cohens_d(d):
    \"\"\"Interpret effect size\"\"\"
    d = abs(d)
    if d < 0.2: return 'negligible'
    elif d < 0.5: return 'small'
    elif d < 0.8: return 'medium'
    else: return 'large'

def interpret_p(p):
    \"\"\"Interpret p-value\"\"\"
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

print('='*60)
print('STATISTICAL SIGNIFICANCE TESTS')
print('='*60)""")

    # Cell 8: Test 1
    add_code("""# Test 1: B0 vs KD2 on SQuAD
b0_squad = df[(df['method'] == 'B0') & (df['task'] == 'squad')]['eval_loss']
kd2_squad = df[(df['method'] == 'KD2') & (df['task'] == 'squad')]['eval_loss']

t_stat, p_value = stats.ttest_ind(b0_squad, kd2_squad)
d = cohens_d(b0_squad, kd2_squad)

print('\\nTest 1: B0 vs KD2 on SQuAD (Eval Loss)')
print('-'*50)
print(f'B0 Mean:  {b0_squad.mean():.4f} +/- {b0_squad.std():.4f}')
print(f'KD2 Mean: {kd2_squad.mean():.4f} +/- {kd2_squad.std():.4f}')
print(f'Difference: {(kd2_squad.mean() - b0_squad.mean()):.4f} ({(kd2_squad.mean() - b0_squad.mean())/b0_squad.mean()*100:.1f}%)')
print(f"\\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f} {interpret_p(p_value)}")
print(f"Cohen's d: {d:.4f} ({interpret_cohens_d(d)})")

if p_value < 0.05:
    winner = 'B0' if b0_squad.mean() < kd2_squad.mean() else 'KD2'
    print(f'\\nResult: {winner} significantly outperforms (p < 0.05)')
else:
    print('\\nResult: No significant difference (p >= 0.05)')""")

    # Cell 9: Test 2
    add_code("""# Test 2: B0 SST-2 vs B0 SQuAD (task difficulty)
b0_sst2 = df[(df['method'] == 'B0') & (df['task'] == 'sst2')]['eval_loss']
b0_squad = df[(df['method'] == 'B0') & (df['task'] == 'squad')]['eval_loss']

t_stat2, p_value2 = stats.ttest_ind(b0_sst2, b0_squad)
d2 = cohens_d(b0_sst2, b0_squad)

print('\\nTest 2: SST-2 vs SQuAD Difficulty (B0 Baseline)')
print('-'*50)
print(f'SST-2 Mean:  {b0_sst2.mean():.4f} +/- {b0_sst2.std():.4f}')
print(f'SQuAD Mean: {b0_squad.mean():.4f} +/- {b0_squad.std():.4f}')
print(f"\\nt-statistic: {t_stat2:.4f}")
print(f"p-value: {p_value2:.6f} {interpret_p(p_value2)}")
print(f"Cohen's d: {d2:.4f} ({interpret_cohens_d(d2)})")""")

    # Cell 10: Stats summary
    add_code("""# Compile all statistical tests
stat_results = [
    {
        'Comparison': 'B0 vs KD2 (SQuAD)',
        'Group 1 Mean': f'{b0_squad.mean():.4f}',
        'Group 2 Mean': f'{kd2_squad.mean():.4f}',
        't-statistic': f'{t_stat:.3f}',
        'p-value': f'{p_value:.4f}',
        'Significance': interpret_p(p_value),
        "Cohen's d": f'{d:.3f}',
        'Effect Size': interpret_cohens_d(d)
    },
    {
        'Comparison': 'SST-2 vs SQuAD (B0)',
        'Group 1 Mean': f'{b0_sst2.mean():.4f}',
        'Group 2 Mean': f'{b0_squad.mean():.4f}',
        't-statistic': f'{t_stat2:.3f}',
        'p-value': f'{p_value2:.4f}',
        'Significance': interpret_p(p_value2),
        "Cohen's d": f'{d2:.3f}',
        'Effect Size': interpret_cohens_d(d2)
    }
]

df_stats = pd.DataFrame(stat_results)
print('\\n' + '='*60)
print('STATISTICAL TESTS SUMMARY')
print('='*60)
display(df_stats)""")

    # Cell 11: Figures header
    add_md("## 3. Publication-Quality Figures")

    # Cell 12: Style setup
    add_code("""# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color palette
COLORS = {
    'B0': '#2ecc71',      # Green
    'KD2': '#3498db',     # Blue
    'sst2': '#e74c3c',    # Red
    'squad': '#9b59b6'   # Purple
}

print('Publication style configured!')""")

    # Cell 13: Figure 1
    add_code("""# Figure 1: Method Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for SQuAD comparison
methods = ['B0 (Baseline)', 'KD2 (Seq-Level KD)']
means = [b0_squad.mean(), kd2_squad.mean()]
stds = [b0_squad.std(), kd2_squad.std()]
colors = [COLORS['B0'], COLORS['KD2']]

bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, 
              edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    ax.annotate(f'{mean:.3f}+/-{std:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add significance annotation
sig_text = f'p = {p_value:.3f} ({interpret_p(p_value)})'
y_max = max(means) + max(stds) + 0.03
ax.plot([0, 0, 1, 1], [y_max-0.01, y_max, y_max, y_max-0.01], 'k-', lw=1.5)
ax.text(0.5, y_max + 0.005, sig_text, ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Evaluation Loss (lower is better)')
ax.set_title('Knowledge Distillation Methods Comparison on SQuAD')
ax.set_ylim(0, y_max + 0.05)

plt.tight_layout()
plt.savefig(FIGS_DIR / 'thesis_fig1_method_comparison.png', dpi=300)
plt.savefig(FIGS_DIR / 'thesis_fig1_method_comparison.pdf')
plt.show()
print('Saved: thesis_fig1_method_comparison.png/pdf')""")

    # Cell 14: Figure 2
    add_code("""# Figure 2: Task Comparison
fig, ax = plt.subplots(figsize=(10, 6))

pivot = df.groupby(['method', 'task'])['eval_loss'].agg(['mean', 'std']).reset_index()

x = np.arange(2)  # sst2, squad
width = 0.35

for i, method in enumerate(['B0', 'KD2']):
    data = pivot[pivot['method'] == method]
    if len(data) > 0:
        tasks = ['sst2', 'squad']
        means = [data[data['task'] == t]['mean'].values[0] if len(data[data['task'] == t]) > 0 else 0 for t in tasks]
        stds = [data[data['task'] == t]['std'].values[0] if len(data[data['task'] == t]) > 0 else 0 for t in tasks]
        
        ax.bar(x + i*width, means, width, yerr=stds, 
               label=method, color=COLORS[method], capsize=5,
               edgecolor='black', linewidth=1, alpha=0.85)

ax.set_xlabel('Task')
ax.set_ylabel('Evaluation Loss (lower is better)')
ax.set_title('Performance by Method and Task')
ax.set_xticks(x + width/2)
ax.set_xticklabels(['SST-2 (Sentiment)', 'SQuAD (QA)'])
ax.legend(title='Method')
ax.set_ylim(0, 0.7)

plt.tight_layout()
plt.savefig(FIGS_DIR / 'thesis_fig2_task_comparison.png', dpi=300)
plt.savefig(FIGS_DIR / 'thesis_fig2_task_comparison.pdf')
plt.show()
print('Saved: thesis_fig2_task_comparison.png/pdf')""")

    # Cell 15: Figure 3
    add_code("""# Figure 3: Seed Variability (Box Plot)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SST-2
ax1 = axes[0]
sst2_data = df[df['task'] == 'sst2']
if len(sst2_data) > 0:
    sns.boxplot(data=sst2_data, x='method', y='eval_loss', ax=ax1, 
                palette=[COLORS['B0']], width=0.5)
    sns.stripplot(data=sst2_data, x='method', y='eval_loss', ax=ax1,
                  color='black', size=8, alpha=0.7)
ax1.set_title('SST-2 (Sentiment Classification)')
ax1.set_xlabel('Method')
ax1.set_ylabel('Evaluation Loss')

# SQuAD
ax2 = axes[1]
squad_data = df[df['task'] == 'squad']
sns.boxplot(data=squad_data, x='method', y='eval_loss', ax=ax2,
            palette=[COLORS['B0'], COLORS['KD2']], width=0.5)
sns.stripplot(data=squad_data, x='method', y='eval_loss', ax=ax2,
              color='black', size=8, alpha=0.7)
ax2.set_title('SQuAD (Question Answering)')
ax2.set_xlabel('Method')
ax2.set_ylabel('Evaluation Loss')

plt.suptitle('Seed Variability Across Training Runs (n=3 per condition)', y=1.02)
plt.tight_layout()
plt.savefig(FIGS_DIR / 'thesis_fig3_variability.png', dpi=300)
plt.savefig(FIGS_DIR / 'thesis_fig3_variability.pdf')
plt.show()
print('Saved: thesis_fig3_variability.png/pdf')""")

    # Cell 16: Figure 4
    add_code("""# Figure 4: Latency Comparison
fig, ax = plt.subplots(figsize=(10, 6))

df_bench['method'] = df_bench['id'].apply(lambda x: x.split('_')[0])
lat_summary = df_bench.groupby('method')['lat_mean'].agg(['mean', 'std']).reset_index()

colors = [COLORS.get(m, '#95a5a6') for m in lat_summary['method']]
bars = ax.bar(lat_summary['method'], lat_summary['mean'], yerr=lat_summary['std'],
              capsize=8, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

for bar, mean, std in zip(bars, lat_summary['mean'], lat_summary['std']):
    ax.annotate(f'{mean:.1f} ms',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Method')
ax.set_ylabel('Inference Latency (ms, lower is better)')
ax.set_title('Inference Latency by Training Method (H100 GPU)')

plt.tight_layout()
plt.savefig(FIGS_DIR / 'thesis_fig4_latency.png', dpi=300)
plt.savefig(FIGS_DIR / 'thesis_fig4_latency.pdf')
plt.show()
print('Saved: thesis_fig4_latency.png/pdf')""")

    # Cell 17: LaTeX header
    add_md("## 4. LaTeX Tables for Thesis")

    # Cell 18: Table 1
    add_code("""# Table 1: Main Results
print('='*70)
print('TABLE 1: Main Experimental Results')
print('='*70)

latex_table1 = '''
\\\\begin{table}[htbp]
\\\\centering
\\\\caption{Experimental Results: Evaluation Loss by Method and Task}
\\\\label{tab:main_results}
\\\\begin{tabular}{llccc}
\\\\toprule
\\\\textbf{Method} & \\\\textbf{Task} & \\\\textbf{Mean} & \\\\textbf{Std} & \\\\textbf{95\\\\% CI} \\\\\\\\
\\\\midrule
'''

for _, row in summary.iterrows():
    latex_table1 += f"{row['method']} & {row['task'].upper()} & {row['mean']:.4f} & {row['std']:.4f} & +/-{row['ci_95']:.4f} \\\\\\\\\\\\\\n"

latex_table1 += '''\\\\bottomrule
\\\\end{tabular}
\\\\end{table}
'''

print(latex_table1)

with open(RESULTS_DIR / 'latex_table1_results.tex', 'w') as f:
    f.write(latex_table1.replace('\\\\\\\\', '\\\\'))
print('\\nSaved: results/latex_table1_results.tex')""")

    # Cell 19: Table 2
    add_code("""# Table 2: Statistical Tests
print('='*70)
print('TABLE 2: Statistical Significance Tests')
print('='*70)

latex_table2 = '''
\\\\begin{table}[htbp]
\\\\centering
\\\\caption{Statistical Significance Tests}
\\\\label{tab:statistics}
\\\\begin{tabular}{lcccc}
\\\\toprule
\\\\textbf{Comparison} & \\\\textbf{t-stat} & \\\\textbf{p-value} & \\\\textbf{Cohen's d} & \\\\textbf{Effect} \\\\\\\\
\\\\midrule
'''

for _, row in df_stats.iterrows():
    latex_table2 += f"{row['Comparison']} & {row['t-statistic']} & {row['p-value']} & {row[\"Cohen's d\"]} & {row['Effect Size']} \\\\\\\\\\\\\\n"

latex_table2 += '''\\\\bottomrule
\\\\end{tabular}
\\\\end{table}
'''

print(latex_table2)

with open(RESULTS_DIR / 'latex_table2_statistics.tex', 'w') as f:
    f.write(latex_table2.replace('\\\\\\\\', '\\\\'))
print('\\nSaved: results/latex_table2_statistics.tex')""")

    # Cell 20: Narrative header
    add_md("## 5. Thesis Narrative Summary")

    # Cell 21: Narrative
    add_code("""# Generate thesis-ready paragraph
b0_squad_mean = b0_squad.mean()
kd2_squad_mean = kd2_squad.mean()
diff_pct = (kd2_squad_mean - b0_squad_mean) / b0_squad_mean * 100

narrative = f'''
================================================================================
THESIS NARRATIVE (Copy-paste ready)
================================================================================

## Results Section Text:

We evaluated two training approaches on a TinyLlama-1.1B student model:
baseline fine-tuning (B0) and sequence-level knowledge distillation (KD2)
using a Qwen-2.5-3B teacher.

**Main Finding:** Contrary to expectations, the baseline approach (B0) achieved
lower evaluation loss ({b0_squad_mean:.4f} +/- {b0_squad.std():.4f}) compared to
KD2 ({kd2_squad_mean:.4f} +/- {kd2_squad.std():.4f}) on the SQuAD question-answering
task. This represents a {abs(diff_pct):.1f}% {'increase' if diff_pct > 0 else 'decrease'}
in loss for the distillation method.

Statistical analysis revealed {'a significant difference' if p_value < 0.05 else 'no statistically significant difference'}
between methods (t = {t_stat:.3f}, p = {p_value:.4f}, Cohen\\'s d = {d:.3f}, {interpret_cohens_d(d)} effect).

**Interpretation:** These results suggest that sequence-level knowledge
distillation may not universally outperform direct fine-tuning, particularly
when (1) the student model has sufficient capacity to learn from gold labels,
(2) the teacher-generated pseudo-labels introduce distribution shift, or
(3) the task does not require complex reasoning beyond pattern matching.

**Task Comparison:** SST-2 sentiment classification achieved substantially
lower loss ({b0_sst2.mean():.4f}) compared to SQuAD ({b0_squad_mean:.4f}),
reflecting the relative complexity of question-answering versus binary
classification (t = {t_stat2:.3f}, p = {p_value2:.4f}, {interpret_cohens_d(d2)} effect size).

================================================================================
'''

print(narrative)

with open(RESULTS_DIR / 'thesis_narrative.txt', 'w') as f:
    f.write(narrative)
print('Saved: results/thesis_narrative.txt')""")

    # Cell 22: Export header
    add_md("## 6. Export Summary")

    # Cell 23: Summary
    add_code("""print('='*70)
print('EXPORT SUMMARY')
print('='*70)

print('\\nDATA FILES:')
print(f'   - {RUNS_DIR}/results.csv')
print(f'   - {RUNS_DIR}/benchmarks.csv')

print('\\nFIGURES (PNG + PDF):')
for f in sorted(FIGS_DIR.glob('thesis_*.png')):
    print(f'   - {f.name}')

print('\\nLATEX TABLES:')
for f in sorted(RESULTS_DIR.glob('latex_*.tex')):
    print(f'   - {f.name}')

print('\\nNARRATIVE:')
print('   - results/thesis_narrative.txt')

print('\\n' + '='*70)
print('THESIS ANALYSIS COMPLETE!')
print('='*70)""")

    # Build notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write
    out_path = Path(__file__).parent.parent / 'notebooks' / '08_thesis_analysis.ipynb'
    with open(out_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f'Created: {out_path}')
    print(f'Size: {out_path.stat().st_size:,} bytes')

if __name__ == '__main__':
    create_notebook()
