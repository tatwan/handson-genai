# Module 03: Overview of Generative AI

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 1 of 3 | **Position:** Module 3 of 10 | **Notebooks:** 3

---

## Overview

Modules 01 and 02 focused on models that *learn from* data — classifiers, regressors, and deep networks that map inputs to labels or predictions. This module marks a fundamental mindset shift: you will now study models that *create* data. Generative models learn the underlying statistical distribution of their training data well enough to draw new, plausible samples from it — synthesizing images, generating text, reconstructing signals, and more.

This distinction matters beyond academic curiosity. Every large language model you will encounter from Module 05 onward, every text-to-image system, and every retrieval-augmented pipeline you build in later modules rests on generative principles introduced here. Understanding how models encode meaning into a compressed latent space, how sampling temperature shapes output diversity, and how encoder-decoder architectures transform representations will give you the vocabulary and intuition to reason about — and debug — modern AI systems confidently.

---

## Where This Module Fits

```
Module 01: ML Foundations
      ↓
Module 02: Deep Learning Primer
      ↓
Module 03: Overview of Generative AI  ← you are here
      ↓
Module 04: NLP — Understanding Language as Data
      ↓
Module 05: Large Language Models (LLMs)
      ↓
Modules 06–10: Prompting · Fine-Tuning · RAG · Optimization · Capstone
```

The three core concepts introduced here — latent space, probabilistic sampling, and encoder-decoder architecture — resurface in every subsequent module. Word embeddings (Module 04), transformer internals (Module 05), fine-tuning mechanics (Module 07), and vector databases for RAG (Module 08) all build directly on what you learn today.

---

## Learning Objectives

By the end of this module you will be able to:

- Explain the difference between discriminative and generative models, and articulate why that distinction matters for practical AI development
- Understand probabilistic sampling and describe how temperature controls the creativity vs. consistency trade-off in generated outputs
- Describe what a latent space is, why compressing data into one is useful, and how to visualize it
- Build and train an Autoencoder on the MNIST dataset, observing how a bottleneck forces the network to learn compact representations
- Explain how a Variational Autoencoder (VAE) extends the standard Autoencoder by mapping inputs to distributions, yielding a structured and sampleable latent space
- Describe the intuition behind diffusion models — progressive noise addition and iterative denoising — and connect guidance scale to classifier-free guidance
- Connect these architectures to production systems such as Stable Diffusion and large language models, understanding which components play which roles

---

## Module Notebooks

### 📓 01 · Introduction to Generative AI

**File:** `01_intro_to_generative_ai.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_03_Generative_AI/01_intro_to_generative_ai.ipynb)

**Concepts covered:**
- Discriminative vs. generative models
- Probabilistic sampling and temperature
- Latent space and what it means to learn a data distribution
- Overview of generative model families: Autoencoders, VAEs, GANs, Transformers, Diffusion Models

**Lab builds:**
- Temperature-controlled word sampler demonstrating how a single parameter shapes output diversity
- Gaussian Mixture Model (GMM) synthetic data generator to build intuition for sampling from distributions
- PCA-based MNIST latent space visualization showing how 784-dimensional images compress into 2D structure

> **Key Insight:** Generative models don't classify data — they learn its distribution and sample from it. Temperature controls how creative (or conservative) that sampling is. A temperature of 0 always picks the most likely next token; high temperature explores the full probability distribution.

---

### 📓 02 · Autoencoders and Variational Autoencoders

**File:** `02_autoencoders.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_03_Generative_AI/02_autoencoders.ipynb)

**Concepts covered:**
- Encoder-decoder architecture and the role of the bottleneck
- Latent code: compressed internal representation
- Denoising Autoencoders: robustness through corruption
- Reparameterization trick: making stochastic sampling differentiable
- KL divergence: regularizing the latent space toward a prior
- Structured latent space and why smoothness enables generation

**Lab builds:**
- Standard Autoencoder (784 → 32 → 784) trained on MNIST with reconstruction quality analysis
- Denoising Autoencoder that learns to recover clean images from corrupted inputs
- Full VAE with 2D latent space visualization and latent manifold grid showing smooth interpolation between digits

> **Key Insight:** The VAE's structured latent space makes smooth generation possible. Mapping inputs to *distributions* (not fixed points) means any sampled point decodes to something plausible. This same idea lives inside Stable Diffusion's image encoder — images are encoded into a latent distribution before diffusion operates on them.

---

### 📓 03 · Diffusion Models

**File:** `03_diffusion_models.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_03_Generative_AI/03_diffusion_models.ipynb)

**Concepts covered:**
- Forward diffusion: progressively adding Gaussian noise over a schedule
- Reverse diffusion: learning to denoise step by step
- Noise schedules: linear, cosine, and their effect on training
- Classifier-free guidance: conditioning on text prompts without a separate classifier
- Latent diffusion: operating in compressed VAE latent space for efficiency
- Text-to-image generation end-to-end pipeline

**Lab builds:**
- NumPy visualization of the forward diffusion process (no GPU required) illustrating the noise schedule
- Text-to-image pipeline with Stable Diffusion using the Hugging Face `diffusers` library
- Controlled experiments varying `guidance_scale`, `num_inference_steps`, `negative_prompt`, and random seeds to build intuition for each parameter

> **Key Insight:** Diffusion models iteratively denoise random noise into coherent images. The `guidance_scale` parameter works via classifier-free guidance — the model runs twice per step (once conditioned on the prompt, once unconditioned) and interpolates between the two predictions. Higher values push harder toward the prompt but reduce variety; lower values allow more creative freedom.

---

## The Conceptual Thread

The three notebooks tell a single story of progressively more powerful generative architectures:

1. **Autoencoders teach compression.** The bottleneck forces the network to discard noise and retain only what is essential about the data. Reconstruction loss trains the encoder and decoder jointly without any labels.

2. **VAEs teach structured compression.** Instead of mapping each input to a fixed point, a VAE maps it to a probability distribution (mean and variance). The reparameterization trick keeps the process differentiable. KL divergence regularizes the latent space into a smooth, continuous manifold — meaning you can sample any point in that space and the decoder will produce something plausible, not garbage.

3. **Diffusion models teach iterative refinement.** Rather than decoding in a single forward pass, diffusion models reverse a gradual noising process over hundreds of steps. Each small step is a tractable denoising task. The result is far higher perceptual quality and fine-grained control than single-pass generation.

Each step solves a clear limitation of the previous approach. This progression explains why diffusion became the dominant paradigm for image generation — and why VAE encoders still live *inside* Stable Diffusion: the pipeline encodes the image with a VAE into a compact latent space, runs the diffusion process entirely in that latent space (much cheaper than pixel space), then decodes back to pixels with the VAE decoder.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 03 | Where it reappears |
|---|---|
| Latent space and compressed representations | Module 04 (word embeddings), Module 08 (vector databases for RAG) |
| Probabilistic sampling and temperature | Module 05 (LLM token decoding), Module 07 (sampling techniques lab) |
| Encoder-decoder architecture | Module 05 (Transformer internals, BERT encoder, GPT decoder) |
| VAE as image encoder/decoder | Module 05 (Stable Diffusion multimodal lab) |
| Representation learning | Module 07 (fine-tuning: what does the model already know?) |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free GPU access, no local setup required)
- **Local:** Python 3.10+ with a virtual environment; notebooks install dependencies in their first cell automatically

### Dependencies

| Library | Used in |
|---|---|
| `tensorflow` / `keras` | Notebooks 01, 02 |
| `scikit-learn` | Notebook 01 |
| `diffusers` + `torch` | Notebook 03 |
| `numpy` + `matplotlib` | All notebooks |

### Run Order

Run the notebooks in sequence: **01 → 02 → 03**. Concepts build on each other — the latent space intuition from Notebook 01 is essential before examining the VAE latent manifold in Notebook 02, and both inform how you reason about the diffusion pipeline in Notebook 03.

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · Introduction to Generative AI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_03_Generative_AI/01_intro_to_generative_ai.ipynb) |
| 02 · Autoencoders and Variational Autoencoders | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_03_Generative_AI/02_autoencoders.ipynb) |
| 03 · Diffusion Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_03_Generative_AI/03_diffusion_models.ipynb) |
