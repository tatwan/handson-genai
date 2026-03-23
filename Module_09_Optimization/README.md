# Module 9: Model Optimization & Deployment

## Overview

Once you have a fine-tuned or pre-trained model that performs well in the lab, the next challenge is making it run *efficiently* in the real world. This module covers four key techniques for reducing model size and inference cost without sacrificing meaningful accuracy — and then benchmarks them head-to-head so you can make informed decisions.

## Why Optimization Matters

A model fine-tuned in Module 8 may be accurate but impractical for production:

| Problem | Impact | Solution |
|---------|--------|----------|
| Slow inference | Poor user experience | Pruning, Quantization |
| Large memory footprint | Expensive GPU instances | Quantization, Distillation |
| High cost per request | Unsustainable at scale | Any of the four techniques |
| Latency spikes | SLA violations in production | Quantization (deterministic) |

> **Key insight:** Most neural networks are *massively overparameterized*. Research consistently shows that 30–90% of weights can be removed or compressed with minimal accuracy loss — especially for inference workloads.

## Learning Objectives

By the end of this module, you will be able to:

- Explain the trade-offs between model size, speed, and accuracy
- Apply **knowledge distillation** to train a smaller student model from a larger teacher
- Apply **unstructured pruning** with `torch.nn.utils.prune` and analyze the sparsity-accuracy trade-off
- Apply **dynamic quantization** (INT8), **half-precision** (FP16), and **4-bit NF4** quantization using `bitsandbytes`
- Understand modern LLM quantization methods: GPTQ and AWQ
- **Benchmark** multiple optimization strategies head-to-head using a recommendation engine
- Choose the right technique for a given deployment constraint (edge, cloud GPU, CPU-only)

## Module Structure

| Notebook | Topic | Key Libraries |
|----------|-------|---------------|
| `01_intro_to_optimization.ipynb` | Overview, decision framework, cost analysis | `transformers`, `torch`, `pandas` |
| `02_knowledge_distillation.ipynb` | Teacher → student training, KL divergence, temperature | `torch`, `transformers`, `datasets` |
| `03_pruning.ipynb` | Unstructured pruning, sparsity analysis, accuracy trade-off | `torch.nn.utils.prune` |
| `04_quantization.ipynb` | INT8, FP16, 4-bit NF4, GPTQ, AWQ | `bitsandbytes`, `transformers`, `torch` |
| `05_benchmarking.ipynb` | Head-to-head comparison, Pareto frontier, recommendation engine | `torch`, `pandas`, `matplotlib` |

## Core Concepts

### 1. Knowledge Distillation
Train a small **student** model to mimic a large **teacher** model's soft probability outputs (not just its hard labels). The student learns the teacher's "dark knowledge" — the relative probabilities assigned to wrong answers, which contain rich information about the problem structure.

```
Teacher (large, accurate) → KL Divergence Loss → Student (small, fast)
```

**Best when:** You need a model specifically optimized for one task and you have training data.

### 2. Pruning
Remove weights that contribute little to the model's output. `torch.nn.utils.prune` supports:
- **Unstructured pruning** (individual weights → sparse tensors) — up to 90% sparsity possible
- **Structured pruning** (entire neurons/heads → actually reduces model dimensions)

**Best when:** You need CPU-friendly deployment or ONNX/TensorRT export.

### 3. Quantization
Reduce the numerical precision of weights and activations:

| Method | Precision | Memory Reduction | Speed | Accuracy Loss |
|--------|-----------|-----------------|-------|---------------|
| Dynamic INT8 | 8-bit | ~4× | Moderate | Minimal |
| FP16 / BF16 | 16-bit | ~2× | Fast (GPU) | Negligible |
| 4-bit NF4 | 4-bit | ~8× | Moderate | Small |
| GPTQ (post-train) | 4-bit | ~8× | Fast | Very small |
| AWQ (activation-aware) | 4-bit | ~8× | Fast | Smallest loss |

**Best when:** You need the fastest path to smaller memory footprint without retraining.

### 4. Benchmarking
Optimization is a trade-off. `05_benchmarking.ipynb` shows how to plot the **Pareto frontier** — the set of solutions where no technique is strictly better than all others across all dimensions simultaneously — and build a recommendation engine based on your deployment priorities.

## Prerequisites

- Module 8 (Fine-Tuning) — you should understand what a fine-tuned model is before optimizing one
- Python 3.10+, PyTorch, HuggingFace `transformers`
- GPU recommended for notebooks 03 and 04 (pruning visualization and 4-bit quantization)
- Google Colab (free tier) is sufficient for all notebooks

## Environment Setup

```bash
# All notebooks install their own dependencies in the first cell using:
uv pip install transformers datasets torch torchvision bitsandbytes accelerate
```

> **Note:** `bitsandbytes` requires CUDA. Quantization notebooks that use 4-bit NF4 must be run on a GPU runtime (Colab T4 or better).

## Connection to Previous and Next Modules

```
Module 8 (Fine-Tuning)
       ↓
   You have a model that's accurate but large

Module 9 (Optimization)     ← You are here
       ↓
   Apply distillation / pruning / quantization
   Benchmark the results
       ↓
Module 10 (Capstone)
   Deploy your optimized, accurate model
```
