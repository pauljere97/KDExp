#!/usr/bin/env python3
"""Create a Colab-compatible notebook for CUDA training."""
import json

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": []
}

def add_md(lines):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(lines):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# Title
add_md([
    "# Knowledge Distillation Experiments - CUDA\n",
    "\n",
    "**Thesis: Knowledge Distillation for LLMs**\n",
    "\n",
    "## Setup\n",
    "1. Runtime > Change runtime type > **GPU** (T4 or better)\n",
    "2. Run all cells in order\n",
    "3. Results saved to Google Drive"
])

# GPU Check
add_code([
    "import torch\n",
    "print(f'CUDA available: {torch.cuda.is_available()}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
    "    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n",
    "else:\n",
    "    raise RuntimeError('No GPU! Runtime > Change runtime type > GPU')"
])

# Mount Drive
add_code([
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "PROJECT_DIR = '/content/drive/MyDrive/thesis-kd'\n",
    "!mkdir -p {PROJECT_DIR}"
])

# Install deps
add_code([
    "!pip install -q transformers>=4.40.0 datasets>=2.18.0 peft>=0.10.0 accelerate>=0.28.0 scipy scikit-learn tqdm matplotlib seaborn pandas"
])

# Setup
add_code([
    "import os, gc, json, time\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "ROOT = Path('/content/kd_exp')\n",
    "for d in ['models', 'runs', 'figures', 'cache']:\n",
    "    (ROOT / d).mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "DEVICE = torch.device('cuda')\n",
    "print(f'Root: {ROOT}')"
])

# Config
add_code([
    "CONFIG = {\n",
    "    'teacher': 'Qwen/Qwen2.5-3B-Instruct',\n",
    "    'student': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',\n",
    "    'train_n': 5000, 'val_n': 1000,\n",
    "    'batch': 4, 'grad_accum': 4, 'epochs': 3, 'lr': 2e-4,\n",
    "    'fp16': True, 'lora_r': 16, 'lora_alpha': 32,\n",
    "    'seeds': [42, 123, 456]\n",
    "}\n",
    "print(CONFIG)"
])

# Step 1
add_md(["## Step 1: Load Datasets"])

add_code([
    "from datasets import load_dataset\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
    "\n",
    "sst2 = load_dataset('glue', 'sst2')\n",
    "squad = load_dataset('squad')\n",
    "\n",
    "sst2_train = sst2['train'].select(range(min(CONFIG['train_n'], len(sst2['train']))))\n",
    "sst2_val = sst2['validation']\n",
    "squad_train = squad['train'].select(range(min(CONFIG['train_n'], len(squad['train']))))\n",
    "squad_val = squad['validation'].select(range(min(CONFIG['val_n'], len(squad['validation']))))\n",
    "\n",
    "print(f'SST-2: {len(sst2_train)} train, {len(sst2_val)} val')\n",
    "print(f'SQuAD: {len(squad_train)} train, {len(squad_val)} val')"
])

add_code([
    "tokenizer = AutoTokenizer.from_pretrained(CONFIG['student'], trust_remote_code=True)\n",
    "if tokenizer.pad_token is None:\n",
    "    tokenizer.pad_token = tokenizer.eos_token\n",
    "\n",
    "def make_sst2(ex):\n",
    "    lab = 'positive' if ex['label'] == 1 else 'negative'\n",
    "    return {'prompt': f\"Classify: {ex['sentence']}\\nSentiment: {lab}\"}\n",
    "\n",
    "def make_squad(ex):\n",
    "    ans = ex['answers']['text'][0] if ex['answers']['text'] else ''\n",
    "    return {'prompt': f\"Context: {ex['context'][:500]}\\nQ: {ex['question']}\\nA: {ans}\"}\n",
    "\n",
    "sst2_train = sst2_train.map(make_sst2)\n",
    "sst2_val = sst2_val.map(make_sst2)\n",
    "squad_train = squad_train.map(make_squad)\n",
    "squad_val = squad_val.map(make_squad)\n",
    "print('Prompts created!')"
])

add_code([
    "def tokenize(ex, max_len=256):\n",
    "    enc = tokenizer(ex['prompt'], truncation=True, max_length=max_len, padding='max_length')\n",
    "    enc['labels'] = enc['input_ids'].copy()\n",
    "    return enc\n",
    "\n",
    "sst2_train_tok = sst2_train.map(lambda x: tokenize(x, 256), batched=True, remove_columns=sst2_train.column_names)\n",
    "sst2_val_tok = sst2_val.map(lambda x: tokenize(x, 256), batched=True, remove_columns=sst2_val.column_names)\n",
    "squad_train_tok = squad_train.map(lambda x: tokenize(x, 512), batched=True, remove_columns=squad_train.column_names)\n",
    "squad_val_tok = squad_val.map(lambda x: tokenize(x, 512), batched=True, remove_columns=squad_val.column_names)\n",
    "print('Tokenization done!')"
])

# Step 2
add_md(["## Step 2: Cache Teacher Outputs"])

add_code([
    "teacher = AutoModelForCausalLM.from_pretrained(CONFIG['teacher'], trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')\n",
    "teacher.eval()\n",
    "teacher_tok = AutoTokenizer.from_pretrained(CONFIG['teacher'], trust_remote_code=True)\n",
    "if teacher_tok.pad_token is None:\n",
    "    teacher_tok.pad_token = teacher_tok.eos_token\n",
    "print(f\"Teacher: {sum(p.numel() for p in teacher.parameters())/1e9:.2f}B\")"
])

add_code([
    "cache_file = ROOT / 'cache' / 'squad_answers.json'\n",
    "if cache_file.exists():\n",
    "    print('Cache exists')\n",
    "else:\n",
    "    answers = []\n",
    "    for i, ex in enumerate(tqdm(squad_train)):\n",
    "        p = ex['prompt'].split('A:')[0] + 'A:'\n",
    "        inp = teacher_tok(p, return_tensors='pt', truncation=True, max_length=448)\n",
    "        inp = {k: v.to(teacher.device) for k, v in inp.items()}\n",
    "        with torch.no_grad():\n",
    "            out = teacher.generate(**inp, max_new_tokens=64, do_sample=False, pad_token_id=teacher_tok.pad_token_id)\n",
    "        a = teacher_tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)\n",
    "        answers.append({'prompt': p, 'answer': a})\n",
    "        if i % 100 == 0: torch.cuda.empty_cache()\n",
    "    with open(cache_file, 'w') as f: json.dump(answers, f)\n",
    "    print(f'Saved {len(answers)}')"
])

add_code([
    "del teacher\n",
    "torch.cuda.empty_cache()\n",
    "gc.collect()\n",
    "print('Teacher unloaded')"
])

# Step 3
add_md(["## Step 3: Training Functions"])

add_code([
    "from transformers import TrainingArguments, Trainer\n",
    "from peft import LoraConfig, get_peft_model, TaskType, PeftModel\n",
    "\n",
    "def load_student():\n",
    "    m = AutoModelForCausalLM.from_pretrained(CONFIG['student'], trust_remote_code=True, torch_dtype=torch.float16)\n",
    "    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'], target_modules=['q_proj','k_proj','v_proj','o_proj'])\n",
    "    m = get_peft_model(m, lora)\n",
    "    m.print_trainable_parameters()\n",
    "    return m.to(DEVICE)\n",
    "\n",
    "def get_args(out, name):\n",
    "    return TrainingArguments(output_dir=str(out), run_name=name, num_train_epochs=CONFIG['epochs'],\n",
    "        per_device_train_batch_size=CONFIG['batch'], gradient_accumulation_steps=CONFIG['grad_accum'],\n",
    "        learning_rate=CONFIG['lr'], fp16=CONFIG['fp16'], logging_steps=50, eval_strategy='epoch',\n",
    "        save_strategy='epoch', save_total_limit=1, load_best_model_at_end=True, report_to='none')\n",
    "\n",
    "print('Ready!')"
])

# Step 4
add_md(["## Step 4: Run Experiments"])

add_code([
    "MODELS = ROOT / 'models'\n",
    "RUNS = ROOT / 'runs'\n",
    "results = []"
])

add_code([
    "print('='*60)\n",
    "print('B0: BASELINE')\n",
    "for seed in CONFIG['seeds']:\n",
    "    for task, tds, vds in [('sst2', sst2_train_tok, sst2_val_tok), ('squad', squad_train_tok, squad_val_tok)]:\n",
    "        rid = f'B0_{task}_s{seed}'\n",
    "        out = MODELS / rid\n",
    "        if (out / 'final').exists(): print(f'Skip {rid}'); continue\n",
    "        print(f'Training {rid}')\n",
    "        torch.manual_seed(seed); np.random.seed(seed)\n",
    "        m = load_student()\n",
    "        t = Trainer(model=m, args=get_args(out, rid), train_dataset=tds, eval_dataset=vds, processing_class=tokenizer)\n",
    "        tr = t.train(); ev = t.evaluate()\n",
    "        t.save_model(str(out / 'final'))\n",
    "        results.append({'id': rid, 'method': 'B0', 'task': task, 'seed': seed, 'eval_loss': ev['eval_loss']})\n",
    "        print(f\"Done: {ev['eval_loss']:.4f}\")\n",
    "        del m, t; torch.cuda.empty_cache(); gc.collect()"
])

add_code([
    "print('='*60)\n",
    "print('KD2: SEQUENCE-LEVEL')\n",
    "with open(ROOT / 'cache' / 'squad_answers.json') as f: ta = json.load(f)\n",
    "from datasets import Dataset\n",
    "kd2_prompts = [x['prompt'] + ' ' + x['answer'] for x in ta]\n",
    "kd2_enc = tokenizer(kd2_prompts, truncation=True, max_length=512, padding='max_length')\n",
    "kd2_ds = Dataset.from_dict({'input_ids': kd2_enc['input_ids'], 'attention_mask': kd2_enc['attention_mask'], 'labels': kd2_enc['input_ids']})\n",
    "print(f'KD2: {len(kd2_ds)} examples')\n",
    "\n",
    "for seed in CONFIG['seeds']:\n",
    "    rid = f'KD2_squad_s{seed}'\n",
    "    out = MODELS / rid\n",
    "    if (out / 'final').exists(): print(f'Skip {rid}'); continue\n",
    "    print(f'Training {rid}')\n",
    "    torch.manual_seed(seed); np.random.seed(seed)\n",
    "    m = load_student()\n",
    "    t = Trainer(model=m, args=get_args(out, rid), train_dataset=kd2_ds, eval_dataset=squad_val_tok, processing_class=tokenizer)\n",
    "    tr = t.train(); ev = t.evaluate()\n",
    "    t.save_model(str(out / 'final'))\n",
    "    results.append({'id': rid, 'method': 'KD2', 'task': 'squad', 'seed': seed, 'eval_loss': ev['eval_loss']})\n",
    "    print(f\"Done: {ev['eval_loss']:.4f}\")\n",
    "    del m, t; torch.cuda.empty_cache(); gc.collect()"
])

add_code([
    "df = pd.DataFrame(results)\n",
    "df.to_csv(RUNS / 'results.csv', index=False)\n",
    "print(df)"
])

# Step 5
add_md(["## Step 5: Benchmarking"])

add_code([
    "def bench(model, n=20):\n",
    "    inp = tokenizer('The quick brown fox', return_tensors='pt').to(DEVICE)\n",
    "    for _ in range(5):\n",
    "        with torch.no_grad(): model.generate(**inp, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)\n",
    "    torch.cuda.synchronize()\n",
    "    lats = []\n",
    "    for _ in range(n):\n",
    "        s = time.perf_counter()\n",
    "        with torch.no_grad(): model.generate(**inp, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)\n",
    "        torch.cuda.synchronize()\n",
    "        lats.append((time.perf_counter() - s) * 1000)\n",
    "    return {'lat_mean': np.mean(lats), 'lat_std': np.std(lats)}\n",
    "\n",
    "benchs = []\n",
    "for d in sorted(MODELS.iterdir()):\n",
    "    if not (d / 'final').exists(): continue\n",
    "    print(f'Bench: {d.name}')\n",
    "    base = AutoModelForCausalLM.from_pretrained(CONFIG['student'], trust_remote_code=True, torch_dtype=torch.float16)\n",
    "    m = PeftModel.from_pretrained(base, str(d / 'final')).merge_and_unload().to(DEVICE).eval()\n",
    "    b = bench(m)\n",
    "    b['id'] = d.name\n",
    "    benchs.append(b)\n",
    "    print(f\"  {b['lat_mean']:.2f} ms\")\n",
    "    del m, base; torch.cuda.empty_cache()\n",
    "pd.DataFrame(benchs).to_csv(RUNS / 'benchmarks.csv', index=False)"
])

# Step 6
add_md(["## Step 6: Generate Figures"])

add_code([
    "import matplotlib.pyplot as plt\n",
    "FIGS = ROOT / 'figures'\n",
    "df = pd.read_csv(RUNS / 'results.csv')\n",
    "dbench = pd.read_csv(RUNS / 'benchmarks.csv')\n",
    "\n",
    "# Method comparison\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "s = df.groupby(['method', 'task'])['eval_loss'].agg(['mean', 'std']).reset_index()\n",
    "x = np.arange(len(s['method'].unique()))\n",
    "w = 0.35\n",
    "for i, t in enumerate(s['task'].unique()):\n",
    "    td = s[s['task'] == t]\n",
    "    ax.bar(x + i*w, td['mean'], w, yerr=td['std'], label=t, capsize=5)\n",
    "ax.set_xlabel('Method'); ax.set_ylabel('Eval Loss'); ax.set_title('KD Comparison')\n",
    "ax.set_xticks(x + w/2); ax.set_xticklabels(s['method'].unique()); ax.legend()\n",
    "plt.tight_layout(); plt.savefig(FIGS / 'method_comparison.png', dpi=300); plt.show()\n",
    "\n",
    "# Latency\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "dbench['method'] = dbench['id'].apply(lambda x: x.split('_')[0])\n",
    "ls = dbench.groupby('method')['lat_mean'].agg(['mean', 'std']).reset_index()\n",
    "ax.bar(ls['method'], ls['mean'], yerr=ls['std'], capsize=5)\n",
    "ax.set_xlabel('Method'); ax.set_ylabel('Latency (ms)'); ax.set_title('Latency')\n",
    "plt.tight_layout(); plt.savefig(FIGS / 'latency.png', dpi=300); plt.show()"
])

# Step 7
add_md(["## Step 7: Save to Google Drive"])

add_code([
    "import shutil\n",
    "DRIVE = Path(PROJECT_DIR) / 'results'\n",
    "DRIVE.mkdir(parents=True, exist_ok=True)\n",
    "for sub in ['runs', 'figures']:\n",
    "    if (ROOT / sub).exists(): shutil.copytree(ROOT / sub, DRIVE / sub, dirs_exist_ok=True)\n",
    "print(f'Saved to: {DRIVE}')\n",
    "print('='*60)\n",
    "print('EXPERIMENT COMPLETE!')\n",
    "print('='*60)"
])

# Write notebook
with open('notebooks/colab_full_experiment.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Colab notebook created successfully!")
