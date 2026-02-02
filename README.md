# Knowledge Distillation for LLMs: Thesis Experiment Suite

A complete, reproducible experiment suite for investigating knowledge distillation methods applied to Large Language Models. Designed for macOS Apple Silicon (MPS) with automatic CPU fallback.

## 📋 Overview

This repository contains all code and notebooks needed to reproduce the experiments from **Chapter 3-4** of the thesis:

- **Tasks:** SST-2 (sentiment classification), SQuAD v1.1 (extractive QA)
- **KD Methods:**
  - **B0:** Baseline fine-tuning (no distillation)
  - **KD1:** Logit-based distillation (soft targets)
  - **KD2:** Sequence-level distillation (teacher-generated outputs)
  - **KD3:** Feature-based distillation (hidden state matching)
- **Models:**
  - Teacher: Qwen2.5-3B-Instruct (fallback) or larger 7B/8B models
  - Student S1: TinyLlama-1.1B-Chat
  - Student S2: ~350M quantized proxy (optional)

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
cd /Users/pjere/Workshop/thesis-exp

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy example and edit with your HuggingFace token
cp .env.example .env

# Edit .env and set HF_TOKEN
```

Required variables in `.env`:
```
HF_TOKEN=your_huggingface_token_here
```

### 3. Run Notebooks

Launch Jupyter and run notebooks **in order** (01 → 07):

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose | Runtime (FAST) |
|----------|---------|----------------|
| 01_environment_setup | Verify setup, create directories | ~1 min |
| 02_data_prep_sst2 | Prepare SST-2 dataset | ~2 min |
| 03_data_prep_squad | Prepare SQuAD dataset | ~3 min |
| 04_teacher_cache_outputs | Cache teacher logits/answers | ~15-30 min |
| 05_train_baseline_and_kd1 | Baseline + logit KD training | ~30-60 min |
| 06_train_kd2_and_kd3 | Sequence + feature KD training | ~30-60 min |
| 07_benchmark_and_plots | Benchmarks, figures, tables | ~10 min |

## 📁 Project Structure

```
thesis-exp/
├── .env                    # Environment variables (HF_TOKEN, model names)
├── .env.example            # Template for .env
├── .gitignore
├── requirements.txt
├── README.md
│
├── configs/
│   └── experiment.yaml     # Main experiment configuration
│
├── src/                    # Python modules
│   ├── __init__.py
│   ├── config.py           # Configuration loader
│   ├── utils_seed.py       # Reproducibility utilities
│   ├── data_sst2.py        # SST-2 data processing
│   ├── data_squad.py       # SQuAD data processing
│   ├── teacher_cache.py    # Teacher output caching
│   ├── kd_losses.py        # KD loss functions
│   ├── trainers.py         # Custom HF trainers
│   ├── bench.py            # Efficiency benchmarking
│   ├── plots.py            # Thesis figure generation
│   ├── stats.py            # Statistical tests
│   └── io.py               # Run registry, CSV/JSON I/O
│
├── notebooks/              # Experiment notebooks
│   ├── 01_environment_setup.ipynb
│   ├── 02_data_prep_sst2.ipynb
│   ├── 03_data_prep_squad.ipynb
│   ├── 04_teacher_cache_outputs.ipynb
│   ├── 05_train_baseline_and_kd1.ipynb
│   ├── 06_train_kd2_and_kd3.ipynb
│   └── 07_benchmark_and_plots.ipynb
│
└── results/                # Generated outputs
    ├── processed_data/     # Tokenized datasets
    ├── teacher_cache/      # Cached teacher outputs
    ├── models/             # Trained model checkpoints
    ├── raw_runs/           # Individual run results
    ├── summary/            # Aggregated tables (CSV)
    └── figures/            # Thesis figures (PNG)
```

## ⚙️ Configuration

### FAST vs FULL Mode

Edit `configs/experiment.yaml`:

```yaml
fast_mode: true   # Quick runs with small subsets (default)
# fast_mode: false  # Full experiments for thesis
```

| Setting | FAST Mode | FULL Mode |
|---------|-----------|-----------|
| SST-2 train | 500 samples | Full (~67k) |
| SQuAD train | 200 samples | Full (~87k) |
| Epochs | 1 | 3 |
| KD1 Grid | T∈{2,4}, α∈{0.3,0.5} | T∈{1,2,4,8}, α∈{0.1,0.3,0.5,0.7} |
| Seeds | [42] | [42, 123, 456] |

### Changing Models

Override in `.env`:
```bash
TEACHER_PRIMARY=meta-llama/Llama-3.1-8B-Instruct
TEACHER_FALLBACK=Qwen/Qwen2.5-3B-Instruct
STUDENT_S1=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 🍎 macOS Apple Silicon Notes

This suite is optimized for MPS (Metal Performance Shaders):

- **fp32 precision** used throughout (fp16/bf16 can be unstable on MPS)
- **Gradient checkpointing** enabled for memory efficiency
- **LoRA/PEFT** for parameter-efficient training
- **Periodic cache clearing** via `torch.mps.empty_cache()`
- **Automatic fallback** to smaller teacher if OOM occurs

### Memory Requirements

| Mode | Recommended RAM | Teacher |
|------|-----------------|---------|
| FAST | 16GB+ | 3B fallback |
| FULL | 32GB+ | 7B/8B primary |

## 📊 Outputs

### Tables (Chapter 4)

Generated in `results/summary/`:
- `table_4_1_main_results.csv` - Main performance comparison
- `table_4_2_kd1_ablation.csv` - Temperature × Alpha grid
- `table_4_3_significance.csv` - Statistical tests (t-test, Cohen's d)
- `benchmarks.csv` - Latency, throughput, memory

### Figures (Chapter 4)

Generated in `results/figures/`:
- `fig_4_1_performance_vs_size.png`
- `fig_4_2_latency.png`
- `fig_4_3_kd_comparison.png`
- `fig_4_4_pareto.png`
- `fig_4_5_kd_gain.png`
- `fig_4_6_memory.png`

## 🔄 Reproducibility

- **Seeds:** Configurable in `experiment.yaml` (default: 42, 123, 456)
- **Idempotent runs:** RunRegistry skips completed experiments
- **Deterministic:** `set_seed()` applied before each training run
- **Version pinning:** See `requirements.txt`

## 🛠️ Troubleshooting

### "MPS backend out of memory"
1. Reduce batch size in notebook training args
2. Enable `fast_mode: true` in config
3. Use smaller fallback teacher
4. Close other applications

### "Model not found"
1. Verify HF_TOKEN in `.env`
2. Check model name spelling
3. Some models require access approval on HuggingFace

### Slow training on MPS
- This is expected; MPS is slower than CUDA for LLMs
- Use FAST mode for development
- Run FULL mode overnight or on cloud GPU

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{pauljere2026kd,
  title={Knowledge Distillation for Efficient Large Language Models},
  author={Paul Jere},
  year={2026},
  school={WSB University}
}
```

## 📄 License

MIT License - See LICENSE file for details.
