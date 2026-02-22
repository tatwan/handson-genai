# Benchmarking Optimization Techniques – Summary

------

## Overview

This notebook provides a **wrap-up comparison** of all model optimization techniques introduced in this module using a standardized benchmarking framework.
It focuses on how pruning and quantization impact latency, model size, and accuracy for a DistilBERT sentiment model on SST-2.​

------

## Learning Objectives

By the end of this notebook, you will be able to:

- Set up a **reproducible benchmarking loop** for multiple model variants.
- Compare **latency, memory, sparsity, and accuracy** across optimization techniques.
- Visualize **speed–accuracy–size trade-offs** with clear plots.
- Use a simple **recommendation heuristic** to choose the best variant for your use case.

------

## Model Variants Benchmarked

All variants start from `distilbert-base-uncased-finetuned-sst-2-english`.

- **Baseline FP32**
- **INT8 Quantized** (dynamic quantization on Linear layers, CPU-only)
- **Pruned 30%** (L1 unstructured on Linear layers)
- **Pruned 50%** (more aggressive pruning)
- **Pruned 30% + INT8** (prune then quantize)

Test set: **100 SST-2 samples** for quick but consistent evaluation.

------

## Metrics and Utilities

The notebook defines reusable helpers for:

- **Model size (MB)**: parameters + buffers converted to MB.
- **Sparsity (%)**: fraction of zero weights across all parameters.
- **Latency (ms)**: mean and std over 50 inference runs (with warmup), device-aware sync.
- **Accuracy (%)**: simple loop over test samples with `argmax` on logits.

It also derives:

- **Speedup** = baseline latency / variant latency.
- **Size reduction (%)** = relative decrease vs baseline.
- **Accuracy drop** = baseline accuracy − variant accuracy.

------

## Core Results

Approximate benchmark results (single SST-2 example, 100-test-sample eval):

- **Baseline FP32**
  - Size: ~255.4 MB
  - Latency: ~5.42 ms
  - Accuracy: 91.0%
- **Quantized INT8**
  - Size: ~91.0 MB
  - Latency: ~21.0 ms (CPU)
  - Accuracy: 92.0%
  - Size reduction: ~64% vs baseline
  - Note: **smaller but slower** on CPU without specialized INT8 hardware.
- **Pruned 30%**
  - Size: ~255.4 MB (unstructured pruning, no size change on disk)
  - Latency: ~3.93 ms
  - Accuracy: 90.0%
  - Speedup: ~1.38×
  - Accuracy drop: ~1%
- **Pruned 50%**
  - Size: ~255.4 MB
  - Latency: ~3.71 ms
  - Accuracy: 84.0%
  - Speedup: ~1.46×
  - Accuracy drop: ~7%
- **Pruned 30% + INT8**
  - Size: ~91.0 MB
  - Latency: ~17.8 ms (CPU)
  - Accuracy: 91.0%
  - Note: **baseline-level accuracy** with quantized, pruned model but CPU latency penalty.

------

## Visualizations

The notebook builds a 2×2 figure to summarize trade-offs:

- **Latency bar chart** with per-bar speedup annotations.
- **Accuracy bar chart** with a baseline reference line.
- **Size bar chart** with inline size-reduction annotations.
- **Speedup vs Accuracy scatter** with bubble size proportional to model size and “Pareto frontier” interpretation.

These plots make it easy for students to see which models are **dominated** (worse in both speed and accuracy) and which sit on the **Pareto frontier**.

------

## Recommendation Engine

A simple function ranks models based on a priority and an accuracy tolerance:

- **Inputs**:
  - `priority`: `"speed"`, `"accuracy"`, `"size"`, or `"balanced"`
  - `accuracy_tolerance`: max allowed drop vs baseline (e.g., 1–2%)
- **Outputs** (for 1% tolerance example):
  - `SPEED`: **Pruned 30%** (fastest within tolerance)
  - `ACCURACY`: **INT8** (highest accuracy)
  - `SIZE`: **INT8** (smallest model)
  - `BALANCED`: **Pruned 30%`** (good speedup, small accuracy drop)[file:6]

This illustrates how to **encode requirements** (e.g., “speed first, but max 1–2% accuracy loss”) into a simple decision rule.