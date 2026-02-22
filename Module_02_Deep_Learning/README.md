# Module 02: Deep Learning Primer

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 1 of 3 | **Position:** Module 2 of 10 | **Notebooks:** 2

---

## Overview

Module 01 showed you *that* machine learning works — models discover patterns from data
rather than following explicit rules. This module shows you *how* — specifically, how
stacking layers of artificial neurons creates systems capable of hierarchical feature
learning that shallow ML and rule-based programming could never match. Every architecture
you encounter in Modules 03–09 (VAEs, Transformers, LLMs, fine-tuned adapters) is a
deep neural network at its core. Understanding what happens inside those layers is what
separates confident practitioners from black-box users.

---

## Where This Module Fits

```
Module 01: ML Foundations
      ↓
Module 02: Deep Learning Primer  ← you are here
      ↓
Module 03: Overview of Generative AI
      ↓
Modules 04–10: NLP · LLMs · Prompting · Fine-Tuning · RAG · Optimization · Capstone
```

The two notebooks in this module introduce the two frameworks you will encounter
throughout the course: TensorFlow/Keras (used in Module 03's VAE lab) and PyTorch
(the backbone of the entire HuggingFace ecosystem used in Modules 05, 07, and 08).

---

## Learning Objectives

By the end of this module you will be able to:

- Explain the artificial neuron: weighted inputs, bias, activation function, and output
- Describe how stacking layers creates hierarchical feature learning and why depth matters
- Understand backpropagation and gradient descent as the universal engine that trains any neural network
- Compare activation functions — ReLU, GELU, sigmoid, softmax — and explain when to use each
- Build, train, and evaluate an image classifier in TensorFlow/Keras on MNIST
- Build the same classifier in PyTorch using the explicit training loop pattern
- Connect hidden-layer representations to the latent space concept in Module 03

---

## Module Notebooks

### 📓 01 · Neural Network Basics (TensorFlow/Keras)

**File:** `01_neural_network_basics.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_02_Deep_Learning/01_neural_network_basics.ipynb)

**Concepts covered:**
- The artificial neuron: inputs, weights, bias, activation function, output
- Activation functions: ReLU, GELU, sigmoid, softmax — and which modern LLMs use
- Gradient descent visualized on a simple loss landscape
- Backpropagation: how gradients flow through layers to update weights
- Building a neural network with TensorFlow/Keras: Sequential API, Dense, Dropout, BatchNormalization
- Reading training curves to diagnose underfitting and overfitting

**Lab builds:**
- Activation function visualization (ReLU, GELU, sigmoid, tanh) with modern LLM context
- Gradient descent animation on a 2D loss landscape
- MNIST classifier: build → compile → train → evaluate with TensorFlow/Keras
- Improved architecture challenge: add BatchNorm, deeper layers, optimized dropout

> **Key Insight:** The hidden layers of a neural network are learnable feature detectors. A hidden layer trained on MNIST does not encode pixel values — it encodes edges, curves, and digit fragments. This idea of **learned intermediate representations** is exactly what the VAE encoder does in Module 03 (compressing images into a latent space) and what BERT's attention layers do in Module 05 (compressing tokens into contextual representations).

---

### 📓 02 · Image Classification with PyTorch

**File:** `02_image_classification_pytorch.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_02_Deep_Learning/02_image_classification_pytorch.ipynb)

**Concepts covered:**
- PyTorch tensors vs. numpy arrays
- `Dataset` and `DataLoader`: the abstraction HuggingFace Trainer uses for all data loading
- `nn.Module`: the base class for every HuggingFace Transformer model
- The explicit training loop: `zero_grad → forward → backward → step`
- `model.train()` vs `model.eval()`: controlling dropout and batch norm behavior
- Side-by-side comparison of TF/Keras and PyTorch APIs for the same task

**Lab builds:**
- MNIST classifier in PyTorch: same architecture as NB01 — differences are purely framework differences
- Explicit training loop with per-epoch loss and validation accuracy tracking
- Prediction visualization with green/red correct/incorrect indicators
- Student challenge: improve the model using `nn.GELU`, `nn.BatchNorm1d`, weight decay

> **Key Insight:** PyTorch's `nn.Module` and `DataLoader` are not just framework abstractions — they are the exact API you will use in Module 07 when fine-tuning HuggingFace Transformer models. The explicit training loop (`zero_grad → backward → step`) is what HuggingFace `Trainer` runs internally. Building it by hand here removes the mystery later.

---

## The Conceptual Thread

A single neuron computes a weighted sum and applies an activation — simple enough to
understand in one equation. Stacking neurons into layers creates a system that can learn
hierarchical features: the first layer detects edges, the second detects shapes, the third
detects objects. This depth is what gives deep learning its power, and backpropagation is
what makes it trainable.

The two notebooks tell the same story in two frameworks. TensorFlow/Keras hides the
training loop behind `model.fit()` — excellent for rapid prototyping and production
serving. PyTorch exposes every step — essential for research, custom training objectives,
and the HuggingFace ecosystem that dominates modern LLM work. Both paths lead to the
same mathematical operations.

The hidden layer representations you produce in both notebooks are latent spaces in
miniature. Module 03 makes this formal: a Variational Autoencoder forces those
representations to follow a probability distribution, making the space smooth enough to
sample from and generate new data. The leap from a 64-dimensional hidden layer to a VAE
latent manifold is conceptually smaller than it appears.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 02 | Where it reappears |
|---|---|
| Hidden layers as learned representations | Module 03 (VAE encoder bottleneck), Module 05 (Transformer attention layers) |
| Activation functions (ReLU, GELU, softmax) | Module 03 (VAE), Module 05 (Transformer FFN and output layer) |
| Backpropagation and gradient descent | Module 07 (fine-tuning: gradients flow only through adapter layers in LoRA) |
| Train/val curves and overfitting | Module 07 (fine-tuning pitfalls), Module 10 (capstone evaluation) |
| TensorFlow/Keras API patterns | Module 03 (VAE and autoencoder labs) |
| PyTorch `nn.Module` and `DataLoader` | Module 07 (HuggingFace Trainer), Module 08 (RAG with sentence-transformers) |
| BatchNormalization | Module 05 (LayerNorm in Transformers — same idea, different normalization axis) |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free GPU access, no local setup required)
- **Local:** Python 3.10+ with a virtual environment; each notebook installs dependencies in its first cell automatically

### Dependencies

| Library | Used in |
|---|---|
| `tensorflow` + `keras` | Notebook 01 |
| `torch` + `torchvision` | Notebook 02 |
| `numpy` + `matplotlib` | Both notebooks |

### Run Order

Run notebooks in sequence: **NB01 → NB02**. NB01 introduces the concepts and the TF/Keras API; NB02 assumes you understand the task and focuses on the PyTorch differences.

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · Neural Network Basics (TensorFlow/Keras) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_02_Deep_Learning/01_neural_network_basics.ipynb) |
| 02 · Image Classification with PyTorch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_02_Deep_Learning/02_image_classification_pytorch.ipynb) |
