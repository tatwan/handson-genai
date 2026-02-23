# Module 08: Fine-Tuning

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 3 of 3 | **Position:** Module 8 of 10 | **Notebooks:** 6

---

## Overview

Module 07 showed you how to retrieve external knowledge and inject it into prompts. This module goes deeper: instead of changing what the model *sees*, fine-tuning changes what the model *knows*. You adapt a pre-trained model's weights to a new domain, making it inherently better at specific tasks rather than relying on prompt engineering alone.

The module builds a complete mental model before touching training code. NB01 establishes the conceptual foundation — why transfer learning works, when to use full fine-tuning vs. LoRA vs. prompt engineering, and how catastrophic forgetting happens. NB04 completes the theory with sampling techniques that control how a trained model generates text. The three hands-on notebooks (NB02–NB03 and NB05) demonstrate fine-tuning at three different scales and API levels: a small BERT classifier on sentiment, a seq2seq T5 model on dialogue summarization, and a small LLM (TinyLlama) with LoRA adapters and 4-bit quantization. NB02-OpenAI shows the managed, API-based alternative where you upload data and OpenAI handles the training.

By the end of this module you understand not just *how* to fine-tune, but *why* you would (or wouldn't) fine-tune compared to prompt engineering, RAG, or in-context learning.

---

## Where This Module Fits

```
Module 07: Retrieval-Augmented Generation (RAG)
      ↓
Module 08: Fine-Tuning  ← you are here
      ↓
Module 09: Optimization & Evaluation
      ↓
Module 10: Capstone
```

RAG gives the model access to external knowledge at inference time. Fine-tuning bakes knowledge directly into the model weights. Both are complementary: a fine-tuned model with RAG is often stronger than either alone. Module 09 introduces the evaluation frameworks you need to measure whether fine-tuning actually helped.

---

## Learning Objectives

By the end of this module you will be able to:

- Explain transfer learning and why pre-trained representations generalize across tasks
- Choose between full fine-tuning, feature extraction, partial fine-tuning, and LoRA based on data size and compute constraints
- Describe catastrophic forgetting and apply mitigation strategies (lower learning rates, LoRA, early stopping)
- Fine-tune a BERT-family model for text classification using HuggingFace `Trainer`
- Fine-tune a T5 seq2seq model for summarization and evaluate with ROUGE metrics
- Configure and apply LoRA with `LoraConfig` and `get_peft_model()`, and understand each parameter
- Apply 4-bit quantization (`BitsAndBytesConfig`) to run large model fine-tuning on consumer hardware
- Use `SFTTrainer` from the `trl` library for supervised fine-tuning of an instruction-following LLM
- Fine-tune a model via the OpenAI API: prepare JSONL training data, upload files, create a job, monitor status, and evaluate the result
- Understand and apply temperature, Top-K, and Top-P sampling to control generation diversity

---

## Module Notebooks

### 📓 01 · Transfer Learning and Fine-Tuning Concepts

**File:** `01_transfer_learning.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/01_transfer_learning.ipynb)

**Concepts covered:**
- Transfer learning: the idea that representations learned on one task generalize to others; early/middle/late layer specialization
- The pre-training revolution (BERT, GPT, T5, LLaMA): how training at massive scale makes fine-tuning cheap
- Fine-tuning strategies: full fine-tuning, feature extraction (frozen), partial fine-tuning, LoRA/Adapters/Prefix-tuning
- Catastrophic forgetting: what it is, why it happens, and mitigation techniques (lower LR, frozen layers, EWC, LoRA)
- Decision tree: matching strategy to dataset size and compute budget
- Cost-performance trade-offs: prompt engineering → prompt-tuning → LoRA → full fine-tuning

**Lab builds:**
- `bert-base-uncased` loaded and parameter count displayed — showing what "pre-trained" means in practice
- Matplotlib visualization of four fine-tuning strategies: which layers are frozen vs. trainable
- Simulated learning rate experiment: catastrophic forgetting at high LR vs. preserved general knowledge at low LR
- Decision tree walkthrough for three example scenarios (500 examples, medical QA; 50K examples, custom task; no pre-trained model available)
- **Knowledge checks:** three self-assessment questions with expandable answers

> **Key Insight:** Transfer learning is economically rational: someone else paid millions of dollars to train GPT-3 on 500B tokens. You pay a few dollars to adapt those weights to your task. Fine-tuning is not training — it is re-steering a vehicle that already knows how to drive.

---

### 📓 02a · Sentiment Analysis with BERT

**File:** `02_sentiment_analysis.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/02_sentiment_analysis.ipynb)

**Concepts covered:**
- Text classification fine-tuning with `distilbert-base-uncased` on the IMDB dataset
- Data exploration: label distribution, text length distribution, truncation impact at 512-token limit
- Tokenization for classification: `AutoTokenizer`, padding/truncation, `DataCollatorWithPadding`
- `TrainingArguments`: learning rate, weight decay, warmup ratio, `fp16`, `eval_strategy`, `load_best_model_at_end`
- Evaluation metrics: accuracy, F1, precision, recall — why accuracy alone is insufficient
- Overfitting detection: plotting validation loss across epochs
- Confusion matrix and classification report: finding systematic errors
- Error analysis: inspecting misclassified examples to understand model failure modes
- Inference pipeline: `pipeline("sentiment-analysis")` for live predictions

**Lab builds:**
- IMDB dataset loaded, balanced sampling (2,000 train / 500 test) for fast classroom iteration
- `DistilBERT` fine-tuned for binary sentiment (POSITIVE/NEGATIVE) with full `Trainer` loop
- `compute_metrics` function using `evaluate` library: accuracy, F1, precision, recall
- Validation loss curve plotted per epoch with overfitting alert
- Confusion matrix heatmap and full classification report
- Error analysis cell: prints 3 misclassified reviews with true vs. predicted labels
- Inference pipeline tested on 4 custom reviews including ambiguous cases
- **Student challenge:** experiment loop template for testing multiple learning rates

> **Key Insight:** Fine-tuning DistilBERT for 3 epochs on 1,800 examples achieves ~85-88% accuracy on IMDB. Training the same architecture from scratch on this data would converge to near-random performance. The difference is entirely in the pre-trained weights — 66M parameters that already "understand" English from Wikipedia and BookCorpus.

---

### 📓 02b · OpenAI Fine-Tuning API

**File:** `02_fine_tuning_openai.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/02_fine_tuning_openai.ipynb)

**Concepts covered:**
- OpenAI's managed fine-tuning API: how it differs from local fine-tuning (you upload data, OpenAI trains)
- JSONL fine-tuning format: the `{"messages": [...]}` structure with `system`/`user`/`assistant` roles
- Dataset preparation: parsing HuggingFace conversation datasets (`[|Human|]`/`[|AI|]` format) into OpenAI format
- Train/validation split for fine-tuning: detecting overfitting during the job
- API lifecycle: `files.create()` → `fine_tuning.jobs.create()` → `jobs.retrieve()` → inference
- Evaluating a fine-tuned model: base model vs. fine-tuned model comparison on held-out examples
- Cleanup: deleting uploaded files and fine-tuned model endpoints to avoid ongoing costs

**Lab builds:**
- `Mohammed-Altaf/medical-instruction-120k` dataset loaded from HuggingFace (medical QA conversations)
- `convert_hf_conversation_to_openai()` parser handling the `[|Human|]`/`[|AI|]` conversation format
- 100-example subset formatted into JSONL: 90 train / 10 validation
- Full API workflow: upload files, create `gpt-3.5-turbo-0125` fine-tuning job, monitor status
- Side-by-side comparison: base model response vs. fine-tuned model response on held-out medical question
- Cleanup cell: `client.models.delete()` and `client.files.delete()` to manage costs

> **Key Insight:** The OpenAI fine-tuning API abstracts away all the infrastructure complexity — you never see a GPU or a training loop. The trade-off is that you cannot inspect the model internals, adjust hyperparameters beyond what the API exposes, or use it without billing. Understanding both the managed API (NB02b) and local fine-tuning (NB02a, NB03, NB05) makes you fluent at both ends of the cost/control spectrum.

---

### 📓 03 · Text Summarization with T5

**File:** `03_summarization.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/03_summarization.ipynb)

**Concepts covered:**
- Encoder-decoder architecture (T5): encoder reads the full input, decoder generates output token by token
- Sequence-to-sequence fine-tuning: why it differs from classification (variable-length output, teacher forcing)
- T5's task prefix pattern: `"summarize: "` prefix that routes the model to the right task head
- `Seq2SeqTrainingArguments` and `Seq2SeqTrainer`: `predict_with_generate=True`, `generation_max_length`
- ROUGE metrics: ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence)
- Token decoding with numpy int64: HuggingFace's Fast Tokenizer (Rust backend) requires explicit Python `int` conversion for predicted token IDs; `-100` padding must be replaced with `pad_token_id`
- Generation strategies in inference: greedy decoding, beam search (`num_beams`), temperature sampling

**Lab builds:**
- `knkarthick/dialogsum` dataset: 1,000 train / 200 validation, dialogue → summary pairs
- Dialogue length analysis: mean/max word counts and compression ratio computed
- `preprocess_function()` adding T5 prefix, tokenizing inputs and labels with separate max lengths
- `compute_metrics()` with fully documented numpy int64 → Python int fix (critical for production use)
- `t5-small` fine-tuned for 3 epochs with ROUGE-L as the best-model metric
- Evaluation table: ROUGE-1, ROUGE-2, ROUGE-L on validation set
- Summarization pipeline tested on 3 held-out test examples + 1 custom workplace dialogue
- Generation comparison: greedy vs. beam search (k=4) vs. temperature sampling side-by-side
- **Student challenge:** framework for adapting the pipeline to CNN/DailyMail news summarization

> **Key Insight:** The numpy int64 token decoding bug (`OverflowError` in the Rust tokenizer backend) is one of the most common silent failures in seq2seq fine-tuning. It only appears when `predict_with_generate=True` is set — which is required for ROUGE evaluation. Understanding the root cause (numpy int64 vs. Python int, `-100` padding convention) prevents hours of debugging in production.

---

### 📓 04 · Sampling Techniques for LLM Output

**File:** `04_sampling_techniques.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/04_sampling_techniques.ipynb)

**Concepts covered:**
- Token-by-token generation: LLMs output probability distributions, not deterministic text
- Greedy decoding: always pick the highest-probability token; deterministic but repetitive
- Temperature scaling: $P(x_i) = \text{softmax}(z_i / T)$; low T → focused, high T → creative
- Top-K sampling: restrict sampling to the K most probable tokens, renormalized
- Top-P (nucleus) sampling: dynamically select the smallest set of tokens covering P probability mass; adaptive to distribution shape
- Beam search: maintain B candidate sequences; balance between quality and diversity
- Task-specific parameter selection: factual QA vs. code vs. summarization vs. creative writing

**Lab builds:**
- Simulated vocabulary probability distribution for all visualizations
- `apply_temperature()`, `sample_with_temperature()`: manual softmax with temperature
- Temperature effect visualization: 4 subplots (T=0.1, 0.5, 1.0, 2.0) showing distribution sharpening
- `top_k_sampling()`: zero-masked probability distribution visualization for K=1, 3, 5
- `top_p_sampling()`: cumulative probability cutoff visualization for P=0.5, 0.8, 0.95
- `visualize_beam_search()`: annotated tree showing 3 active beams across generation steps
- OpenAI API comparisons: `gpt-4o-mini` sampled with different temperature/top_p for factual, code, and creative tasks
- Decision guide table: recommended temperature and top_p ranges for 6 task types
- **Student exercises:** product description sweet spot, Top-K vs Top-P diversity, customer support chatbot settings

> **Key Insight:** Temperature and Top-P are complementary, not interchangeable. Temperature rescales the entire distribution; Top-P truncates it after a probability threshold. Using both simultaneously can produce conflicting effects — a high temperature makes rare tokens more likely, while a low Top-P cuts them out. The right combination depends on whether you want controlled diversity (moderate both) or maximum creativity (high temperature + high Top-P).

---

### 📓 05 · Fine-Tuning an LLM with LoRA (Healthcare QA)

**File:** `05_Fine_Tuning_LLM_Healthcare.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/05_Fine_Tuning_LLM_Healthcare.ipynb)

**Concepts covered:**
- LoRA mechanics: low-rank decomposition matrices added to frozen attention projections (`q_proj`, `k_proj`, `v_proj`); only 0.1-0.3% of parameters trained
- LoRA hyperparameters: `r` (rank), `lora_alpha` (scaling, typically `2*r`), `lora_dropout`, `target_modules`, `task_type`
- 4-bit quantization: `BitsAndBytesConfig` with `nf4` quantization type and `bfloat16` compute dtype; reduces VRAM by ~75%
- `prepare_model_for_kbit_training()`: enables gradient checkpointing and layer normalization casts for quantized training
- `get_peft_model()`: wraps the quantized base model with LoRA adapter layers
- `SFTTrainer` from `trl`: supervised fine-tuning with `formatting_func` for prompt templating, `packing=False` for instruction data
- `SFTConfig`: `paged_adamw_8bit` optimizer, `bf16=True`, `gradient_accumulation_steps`
- Post-training inference: `gradient_checkpointing_disable()`, `eval()`, `use_cache=True`
- Saving and reloading LoRA adapters: `save_pretrained()` saves only the adapter files (few MB); `PeftModel.from_pretrained()` reloads
- Merging and export: `merge_and_unload()` merges LoRA weights into base model for GGUF conversion
- Ollama deployment: convert merged model to GGUF with `llama.cpp`, create a Modelfile, deploy locally

**Lab builds:**
- `keivalya/MedQuad-MedicalQnADataset`: 1,000-example subset of 16,407 medical QA pairs
- `create_prompt_single()`: `### Question: ... ### Answer: ...` prompt template
- Base `TinyLlama-1.1B-Chat-v1.0` tested before fine-tuning on diabetes symptoms question
- `BitsAndBytesConfig` for 4-bit NF4 quantization with bfloat16 compute
- `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj"])`
- `model.print_trainable_parameters()`: displays 3M/1.1B trainable (0.28%)
- `SFTTrainer` training loop: 1 epoch, batch size 4, gradient accumulation 4, `paged_adamw_8bit`
- Before/after comparison: base model response vs. fine-tuned response on held-out medical question
- LoRA adapter save/reload: `PeftModel.from_pretrained()` with quantized base model
- Full GGUF export and Ollama deployment guide: llama.cpp conversion, quantization, Modelfile, `ollama create`

> **Key Insight:** LoRA's elegance is that the original model weights are *never modified*. The adapter matrices are added on top, trained, and can be swapped out. This means you can maintain one base model on disk and dozens of specialized adapters, each a few megabytes. The `get_peft_model()` call in training and the `PeftModel.from_pretrained()` call in inference are mirror operations — understanding both makes you able to productionize LoRA-based models confidently.

---

## The Conceptual Thread

The six notebooks tell a coherent story that ends with a deployed fine-tuned LLM:

1. **Concepts (NB01)** establish *why* fine-tuning is possible (transferred representations) and *when* it's appropriate (vs. prompt engineering or RAG). Without this foundation, fine-tuning appears as a black box.

2. **Sentiment analysis (NB02a)** is the simplest case: a pre-trained encoder, frozen or fine-tuned, with a classification head. The full `Trainer` loop, evaluation metrics, and error analysis skills learned here carry into every subsequent fine-tuning task.

3. **OpenAI API (NB02b)** shows the managed alternative: no local GPU, no hyperparameters, just data formatting + API calls. Understanding both ends of the spectrum (managed vs. local) is essential for making architectural decisions in production.

4. **Summarization (NB03)** introduces seq2seq fine-tuning, ROUGE evaluation, and the critical token decoding fix. The same pattern (encoder-decoder + generation) applies to translation, question answering, and instruction following.

5. **Sampling techniques (NB04)** explain how trained models produce text. A fine-tuned model without properly configured sampling often produces repetitive or incoherent output. Knowing temperature, Top-K, and Top-P turns model deployment from guesswork into engineering.

6. **LLM fine-tuning with LoRA (NB05)** combines everything: quantization to fit a 1.1B model on free hardware, LoRA to train only 0.28% of parameters, and `SFTTrainer` to handle the training loop. The Ollama export section closes the loop from training to local deployment.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 08 | Where it reappears |
|---|---|
| Transfer learning and pre-trained models | Module 09 (evaluation: measuring how much a model improved) |
| ROUGE metrics | Module 09 (automated evaluation pipelines) |
| `Trainer` / `TrainingArguments` patterns | Module 09 (training with callbacks and logging) |
| LoRA adapter architecture | Module 10 (Capstone: choosing efficient fine-tuning for domain tasks) |
| Sampling parameters (temperature, top_p) | Module 09 (generation-based evaluation), Module 10 (agent generation settings) |
| OpenAI fine-tuning lifecycle | Module 10 (Capstone: domain-adapted models in agent pipelines) |
| JSONL training data format | Module 09 (few-shot evaluation data), Module 10 (Capstone data preparation) |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free GPU access for NB02a, NB03; free T4 GPU for NB05)
- **Local:** Python 3.10+ with a virtual environment and CUDA-enabled GPU recommended for NB05; NB01 and NB04 run on CPU
- **Note:** NB05 requires approximately 6-8 GB VRAM for TinyLlama 4-bit training; Google Colab T4 (15 GB VRAM) works well

### API Keys Required

| Notebook | Service | Key name | Where to get it |
|---|---|---|---|
| NB02b (OpenAI fine-tuning) | OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) — billing required |
| NB04 (sampling comparisons) | OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

NB01, NB02a, NB03, and NB05 require **no API keys** — they use HuggingFace public datasets and models.

Store keys as **Colab Secrets** (key icon in sidebar) or in a local `.env` file.

### Dependencies

| Library | Used in |
|---|---|
| `transformers` | All notebooks |
| `datasets` | NB02a, NB02b, NB03, NB05 |
| `evaluate` | NB02a, NB03 |
| `torch` | NB01, NB02a, NB03, NB04, NB05 |
| `accelerate` | NB02a, NB03, NB05 |
| `peft` | NB05 |
| `trl` | NB05 |
| `bitsandbytes` | NB05 |
| `rouge-score` | NB03 |
| `nltk` | NB03 |
| `scikit-learn`, `seaborn` | NB02a |
| `openai` | NB02b, NB04 |
| `python-dotenv` | NB02b, NB04 |

### Run Order

Run notebooks in sequence: **NB01 → NB02a → NB02b → NB03 → NB04 → NB05**. NB01 builds the conceptual foundation that makes the hands-on notebooks understandable. NB02a and NB03 introduce the `Trainer` API at manageable scale. NB02b shows the API-managed alternative. NB04 rounds out the generation theory. NB05 brings it all together at the LLM scale.

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · Transfer Learning Concepts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/01_transfer_learning.ipynb) |
| 02a · Sentiment Analysis (BERT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/02_sentiment_analysis.ipynb) |
| 02b · OpenAI Fine-Tuning API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/02_fine_tuning_openai.ipynb) |
| 03 · Text Summarization (T5) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/03_summarization.ipynb) |
| 04 · Sampling Techniques | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/04_sampling_techniques.ipynb) |
| 05 · LLM Fine-Tuning with LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_08_Fine_Tuning/05_Fine_Tuning_LLM_Healthcare.ipynb) |
