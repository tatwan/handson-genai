# Sentiment Analysis with IMDB

This notebook walks through building a complete **binary sentiment classifier** on the IMDB movie reviews dataset using **DistilBERT** and the Hugging Face ecosystem. It emphasizes not just model training, but also **metrics, diagnostics, and error analysis**.

------

## 1. Learning Goals

By working through this notebook, students will learn how to:

- Handle large text datasets and inspect length distributions.
- Build proper train/validation/test splits for fair evaluation.
- Use multiple evaluation metrics: **accuracy, precision, recall, F1**.
- Detect **overfitting** from loss curves and validation behavior.
- Perform **error analysis** via confusion matrices and misclassified examples.
- Run **interactive inference** with a sentiment-analysis pipeline.



------

## 2. Dataset Overview

This notebook uses the **IMDB movie review dataset** as a **binary sentiment** task: positive vs negative.

Key properties:

- Large train/test splits (25k+ reviews each in the full dataset).
- Balanced labels (roughly 50% positive, 50% negative).
- Long, free-form text reviews, which stress test sequence length handling.

Students will:

- Inspect label distribution to confirm balance.
- Analyze text length statistics (min, max, mean, median) and visualize the distribution.
- Understand why BERT-family models’ **token limits** (e.g., 256–512 tokens) cause truncation and why that matters.

------

## 3. Notebook Structure

## 3.1 Data Loading and Exploration

- Load IMDB with `datasets.load_dataset("imdb")`.
- Print dataset structure and counts for train/test splits.
- Compute label distribution (positive vs negative).
- Compute and plot text length distribution to understand truncation risk.
- Show sample positive and negative reviews (first 500 characters) to build intuition about the data.

------

## 3.2 Train/Validation/Test Splits

To keep training fast in class/demo settings, the notebook uses **subsets** of the full IMDB dataset:

- Choose `train_size` and `test_size` (e.g., 2000 train, 500 test).
- Shuffle and select balanced subsets.
- Create a validation split from the training subset using `train_test_split` with a 90/10 ratio.

------

## 3.3 Tokenization and Model Setup

- Use `distilbert-base-uncased` tokenizer and model.
- Tokenize text with padding, truncation, and `max_length=256` to handle long IMDB reviews.
- Initialize `AutoModelForSequenceClassification` with 2 labels and appropriate `id2label`/`label2id` mappings.
- Print the total number of model parameters to give students a sense of scale.

------

## 3.4 Metrics: Beyond Accuracy

The notebook defines and explains four metrics:

- **Accuracy**: Overall correctness; fine for balanced data.
- **Precision**: “Quality” of positive predictions.
- **Recall**: “Coverage” of actual positives.
- **F1**: Harmonic mean of precision and recall.

It also gives **domain examples**:

- Spam detection: prioritize **precision** (avoid flagging real emails as spam).
- Fraud detection: prioritize **recall** (catch as many fraudulent transactions as possible).

Implementation uses `evaluate` to compute accuracy, precision, recall, and F1 inside a `compute_metrics` function for the Trainer.

------

## 3.5 Training Configuration and Execution

The notebook uses Hugging Face `Trainer` with:

- `num_train_epochs` (e.g., 3).
- `per_device_train_batch_size` (adjustable to avoid OOM).
- `learning_rate` in a typical fine-tuning range (e.g., 2e-5).
- `weight_decay` and `warmup_ratio` for stable training.
- `eval_strategy="epoch"` and `load_best_model_at_end=True`.
- `metric_for_best_model="f1"` to align selection with a robust metric.



------

## 3.6 Training Diagnostics: “Is My Model Learning?”

After training, the notebook extracts `trainer.state.log_history` and plots **validation loss per epoch**.

Students learn to interpret the chart:

- Good behavior: train and validation loss both decrease.
- Overfitting: validation loss increases while train loss continues decreasing.
- Motivation for **early stopping** when validation loss plateaus or rises.



------

## 3.7 Evaluation: Metrics and Benchmarks

The notebook evaluates on the held-out test subset and prints:

- Accuracy
- F1
- Precision
- Recall

It then contextualizes performance:

- Random guessing ≈ 50%.
- Simple keyword baselines ≈ 70–75%.
- Fine-tuned DistilBERT on subsets ≈ mid-80s.
- SOTA IMDB models (with full data and larger models) ≈ 95%+.

------

## 3.8 Confusion Matrix and Classification Report

To dig deeper into performance, the notebook:

- Uses `trainer.predict` to get logits and labels.
- Builds a confusion matrix with `sklearn.metrics.confusion_matrix`.
- Visualizes it with a Seaborn heatmap.
- Prints a `classification_report` showing precision, recall, and F1 per class.

------

## 3.9 Error Analysis

The notebook identifies misclassified reviews and prints a few examples along with their true and predicted labels.

------

## 3.10 Interactive Inference

Using the `pipeline("sentiment-analysis")` interface, the notebook runs inference on a small set of hand-crafted example reviews.

For each review, it prints:

- Predicted label.
- Confidence score.
- A short snippet of the input.

