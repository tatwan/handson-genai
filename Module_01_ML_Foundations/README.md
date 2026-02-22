# Module 01: Foundations of AI and Machine Learning

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 1 of 3 | **Position:** Module 1 of 10 | **Notebooks:** 1

---

## Overview

Modules 02–10 are built on a single assumption: you understand how machine learning
works. This module establishes that foundation. Before studying models that generate
images, write code, or answer questions, you need to understand how models *learn* —
from data, not from hand-written rules. The vocabulary, mental models, and hands-on
workflow you build here resurface in every subsequent module.

---

## Where This Module Fits

```
Module 01: ML Foundations  ← you are here
      ↓
Module 02: Deep Learning Primer
      ↓
Module 03: Overview of Generative AI
      ↓
Modules 04–10: NLP · LLMs · Prompting · Fine-Tuning · RAG · Optimization · Capstone
```

Modules 01 and 02 are the only modules that focus on how models learn rather than what
they can do. Every large language model, every diffusion pipeline, and every fine-tuning
experiment in later modules rests on the gradient descent, train/test split, and
evaluation principles introduced here.

---

## Learning Objectives

By the end of this module you will be able to:

- Distinguish rule-based programming from machine learning and articulate when each is appropriate
- Describe supervised and unsupervised learning with concrete real-world examples
- Walk through the full ML development lifecycle: data collection → preprocessing → training → evaluation
- Implement and evaluate a scikit-learn classifier using train/test splits and standard metrics
- Explain overfitting and recognize it from the gap between training and validation performance
- Interpret evaluation metrics — accuracy, precision, recall, F1 — and choose the right one for a task
- Connect these ML fundamentals to generative AI: every LLM is an ML model that learned from data

---

## Module Notebooks

### 📓 01 · Introduction to ML Concepts

**File:** `01_intro_to_ml_concepts.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_01_ML_Foundations/01_intro_to_ml_concepts.ipynb)

**Concepts covered:**
- Rule-based programming vs. machine learning — same problem, two philosophies
- Supervised learning (labeled data) vs. unsupervised learning (structure discovery)
- The ML development lifecycle: split → preprocess → train → evaluate → iterate
- Overfitting, underfitting, and the bias-variance tradeoff
- Evaluation metrics: accuracy, precision, recall, F1-score, classification report
- Why these fundamentals matter for every generative AI system

**Lab builds:**
- Rule-based temperature classifier contrasted with a learned classifier (same task, different approach)
- Iris dataset classifier with scikit-learn: train/test split, feature scaling, logistic regression, full evaluation
- Polynomial regression overfitting demo: degree 1 vs. 4 vs. 15, visualizing the memorization problem
- Breast cancer student exercise: end-to-end classification pipeline from scratch

> **Key Insight:** Machine learning doesn't follow rules — it discovers them from data. The exact same workflow (prepare data, train on labeled examples, evaluate on held-out data, iterate) governs a simple Iris classifier and a 70-billion-parameter language model. Understanding this lifecycle deeply is what makes every subsequent module legible.

---

## The Conceptual Thread

Rule-based thinking gave us programmable systems that do exactly what we specify — but
break the moment reality doesn't match the rules. Machine learning inverts the approach:
instead of specifying the rules, we specify the examples and let the model find the
pattern. That shift — from explicit logic to statistical pattern recognition — is the
foundational idea behind every model in this course.

The ML lifecycle introduced here (split your data, preprocess consistently, train on the
training set, evaluate only on the held-out test set, diagnose overfitting, iterate) is
not specific to scikit-learn or Iris flowers. It is the universal workflow for building
any learning system — from a logistic regression to a fine-tuned LLM.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 01 | Where it reappears |
|---|---|
| Supervised learning (labeled examples → predictions) | Module 07 (fine-tuning on labeled instruction data) |
| Unsupervised and self-supervised learning (no external labels required) | Module 03 (VAE trained with reconstruction loss — the model reconstructs its own input), Module 05 (LLMs pre-trained via next-token prediction) |
| Train/test split and evaluation metrics | Module 07 (fine-tuning evaluation), Module 10 (capstone project) |
| Overfitting and regularization | Module 07 (fine-tuning pitfalls: catastrophic forgetting, overfitting to small datasets) |
| ML development lifecycle | Every module — the loop never changes, only the model and data do |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free GPU access, no local setup required)
- **Local:** Python 3.10+ with a virtual environment; the notebook installs dependencies in its first cell automatically

### Dependencies

| Library | Used in |
|---|---|
| `scikit-learn` | Notebook 01 |
| `numpy` + `matplotlib` | Notebook 01 |

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · Introduction to ML Concepts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_01_ML_Foundations/01_intro_to_ml_concepts.ipynb) |
