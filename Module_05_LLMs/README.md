# Module 05: Large Language Models (LLMs)

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 2 of 3 | **Position:** Module 5 of 10 | **Notebooks:** 5

---

## Overview

Module 04 taught you how language becomes numbers — tokens and dense semantic vectors. This module shows you what happens when you scale those ideas to billions of parameters, train on the entire internet, and expose the result through a simple API call.

The module is structured as a deliberate progression: you start with the API layer — sending prompts and receiving responses using three different backends through the same Python client — then pull back the curtain to understand what is happening inside those models. You work with the Hugging Face ecosystem, which gives you access to the raw model weights rather than an opaque API. You then go deep on the two foundational architectures, BERT and GPT, understanding why one reads bidirectionally and the other generates left-to-right. You then wrap everything in a Gradio interface to build a shareable, deployable app in minutes. The module closes with multimodal models — GPT-4o-mini and Gemini 2.0 Flash — extending the text-based workflows you have built to images.

By the end of this module you will have called production LLM APIs, run a model locally, built a UI, and written code that can answer questions about an image. Every notebook is designed to run on both Google Colab (no local setup required) and a local Python environment.

---

## Where This Module Fits

```
Module 01: ML Foundations
      ↓
Module 02: Deep Learning Primer
      ↓
Module 03: Overview of Generative AI
      ↓
Module 04: NLP — Understanding Language as Data
      ↓
Module 05: Large Language Models (LLMs)  ← you are here
      ↓
Modules 06–10: Prompting · Fine-Tuning · RAG · Optimization · Capstone
```

Module 04 gave you the representation layer — tokens and embeddings. This module gives you the model layer built on top of that representation. The two together form the conceptual foundation for everything in Modules 06–10: prompt engineering, fine-tuning, RAG, and the capstone project all assume you understand both what an embedding is and how an LLM generates tokens from them.

---

## Learning Objectives

By the end of this module you will be able to:

- Use the OpenAI Python client to call three completely different LLM backends — OpenAI cloud, Ollama local, and Gemini cloud — by changing only `base_url` and `api_key`
- Run a local open-source LLM with Ollama and explain the practical trade-offs (cost, latency, data privacy) versus cloud APIs
- Use the Hugging Face `pipeline()` abstraction to run inference for text classification, NER, question answering, summarization, translation, text generation, and image classification without writing model code
- Explain the architectural difference between encoder-only models (BERT) and decoder-only models (GPT), and know which to use for which tasks
- Describe how Masked Language Modeling trains BERT and how causal language modeling trains GPT
- Control GPT text generation randomness using temperature and explain why lower temperature produces more deterministic output
- Identify model bias risks in deployed NLP systems and articulate mitigation strategies
- Build an interactive web UI for any NLP model using Gradio `Interface` and `Blocks`, and deploy it to Hugging Face Spaces
- Send images to GPT-4o-mini and Gemini 2.0 Flash for captioning, visual question answering, and document analysis
- Choose between OpenAI and Gemini for vision tasks based on cost, context window, and API format requirements

---

## Module Notebooks

### 📓 01 · LLM APIs: OpenAI, Ollama, and Gemini

**File:** `01_openai_ollama.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/01_openai_ollama.ipynb)

**Concepts covered:**
- OpenAI Python SDK: `chat.completions.create`, temperature, `max_tokens`, streaming with `display(Markdown(...))`
- Ollama: installing and serving local open-source models (Qwen, Llama), the `ollama` Python library
- OpenAI API compatibility: how a single `OpenAI` client object routes to three different backends by changing `base_url`
- Gemini via Google AI Studio: free-tier access using `base_url="https://generativelanguage.googleapis.com/v1beta/openai/"` — no new SDK required
- Streaming responses: rendering token-by-token output with `IPython.display`
- Environment setup patterns for both Google Colab (Secrets) and local `.env` files

**Lab builds:**
- First OpenAI API call with `gpt-4o-mini`; temperature and token limit experiments
- Streaming response with live Markdown rendering in the notebook
- Ollama server setup in Colab; pulling and querying `qwen3:4b` locally
- Gemini 2.0 Flash via OpenAI compatibility: same code, different `base_url`
- Gemini streaming — identical pattern to OpenAI
- Provider comparison table: OpenAI / Ollama / Gemini side-by-side on cost, hosting, and `base_url`
- **Activity:** Build an AI email assistant that drafts professional emails with configurable tone

> **Key Insight:** The OpenAI API format has become the *de facto* compatibility standard for LLM APIs. The same Python client and the same code patterns — with only a `base_url` swap — work against OpenAI's cloud, a local Ollama server, and Google's Gemini models. This abstraction is how production AI systems are architected: the provider is hidden behind a stable interface.

---

### 📓 02 · The Hugging Face Ecosystem

**File:** `02_huggingface_tour.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/02_huggingface_tour.ipynb)

**Concepts covered:**
- The Hugging Face Hub: a model zoo of 500,000+ pre-trained models, datasets, and spaces
- The `pipeline()` abstraction: one function that handles tokenization, model inference, and post-processing for any task
- NLP tasks covered: sentiment analysis, named entity recognition (NER), question answering, summarization, translation, text generation
- Multimodal tasks: image classification, image segmentation, text-to-speech, text-to-music generation
- Under the hood: how `AutoTokenizer` and `AutoModel` convert text to token IDs and feed them into the model
- Hugging Face authentication: HF token setup for rate-limit avoidance and gated model access (optional for public models)

**Lab builds:**
- Sentiment analysis pipeline with `distilbert-base-uncased-finetuned-sst-2-english`
- Named entity recognition on real business text
- Extractive question answering over a paragraph context
- Abstractive summarization with BART
- English → French translation
- GPT-2 text generation
- Image classification and segmentation pipelines
- Text-to-speech synthesis
- **Under the Hood demo:** manually tokenizing text with `AutoTokenizer`, inspecting token IDs, decoding back to subwords

> **Key Insight:** The `pipeline()` function is doing the same three-step process your tiktoken + embeddings code did in Module 04 — tokenize, run through the model, decode — but it wraps a full transformer in a single line. Every model on the Hugging Face Hub uses this interface, giving you instant access to thousands of task-specific models without writing model code.

---

### 📓 03 · BERT vs GPT: Inside the Two Foundational Architectures

**File:** `03_bert_gpt.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/03_bert_gpt.ipynb)

**Concepts covered:**
- **BERT (Bidirectional Encoder Representations from Transformers):** encoder-only, bidirectional context, Masked Language Modeling (MLM) pre-training, fine-tuning for classification tasks
- **GPT (Generative Pre-trained Transformer):** decoder-only, causal (left-to-right) attention, next-token prediction, autoregressive generation
- **Encoder-decoder hybrids:** combining BERT-like understanding with GPT-like generation for constrained tasks (summarization, translation)
- Temperature parameter: how scaling logits before softmax controls randomness in generation
- Model bias: how training data bias propagates to model outputs; real-world consequences of deployment; mitigation approaches (data curation, bias detection, fine-tuning, human oversight)

**Lab builds:**
- BERT Masked Language Modeling: predicting `[MASK]` tokens in context (`"The capital of France is [MASK]"`) — observing ranked semantic alternatives
- DistilBERT sentiment classifier with confidence scores
- GPT-2 text generation with temperature variation (0.3, 0.7, 1.0, 1.5): observing the creativity/determinism trade-off
- Encoder-decoder summarization with BART
- **Bias investigation:** prompting the model with occupational sentences and analyzing distributional bias in predictions

> **Key Insight:** BERT and GPT are not competing models — they are complementary architectures optimized for different goals. BERT reads the entire sentence at once to understand it (fill-in-the-blank training). GPT reads left-to-right to predict what comes next (completion training). Modern LLMs like GPT-4 are scaled-up decoder models; embedding models like `text-embedding-3-small` are encoder models. Knowing the distinction tells you which architecture to reach for.

---

### 📓 04 · Building UIs with Gradio

**File:** `04_gradio_ui.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/04_gradio_ui.ipynb)

**Concepts covered:**
- Gradio `Interface`: three-line minimum to turn any Python function into a web UI
- Gradio components: `Textbox`, `Label`, `Slider`, `Dropdown`, `Image`, and `Audio` input/output types
- Gradio `Blocks`: composing multi-tab, multi-section UIs for complex applications
- Theming and styling: custom themes, descriptions, and example inputs
- Shareable links: `launch(share=True)` creates a public URL in seconds — no server or deployment needed
- Hugging Face Spaces: deploying Gradio apps for free persistent hosting
- Automatic REST API generation: every Gradio app exposes a `/predict` endpoint

**Lab builds:**
- Minimal sentiment analysis UI: one `gr.Interface` wrapping a DistilBERT pipeline
- Enhanced UI with confidence label display and probability breakdown
- BART summarization interface with multi-line text input
- Multi-tab `gr.Blocks` app combining sentiment analysis and summarization in one UI
- CUDA-aware device selection: automatic GPU detection for local environments
- **Student Challenge:** design and build a custom Gradio demo for a task of the student's choosing

> **Key Insight:** Gradio collapses the gap between a working model and a shareable product. The same three-line wrapper works for text, images, audio, and video — and automatically generates a REST API that any downstream system can call. Combined with Hugging Face Spaces, you go from a notebook cell to a deployed URL in minutes.

---

### 📓 05 · Multimodal AI: Vision-Language Models

**File:** `05_multimodal_models.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/05_multimodal_models.ipynb)

**Concepts covered:**
- Vision-language models: extending the LLM input from text tokens to image patches + text tokens
- Leading models landscape (2026): GPT-4o-mini, Gemini 2.0 Flash, Claude Vision, LLaVA
- Image input formats: URL references, base64-encoded bytes (OpenAI format), raw bytes via `Part.from_bytes()` (Gemini format)
- GPT-4o-mini via OpenAI SDK: `image_url` content type in chat messages for captioning, VQA, OCR, and multi-image comparison
- Gemini 2.0 Flash via `google-genai` SDK: `Part.from_bytes()` for image input, free-tier access
- Cost optimization: `detail` parameter, model selection, image resizing
- Real-world use cases: accessibility alt-text generation, e-commerce product analysis, content moderation

**Lab builds:**
- Image display helper with `PIL` + `matplotlib` for URLs and local paths
- GPT-4o-mini image captioning from URL
- Visual question answering: four targeted questions on the same image
- Local image analysis via base64 encoding
- Document/chart understanding and OCR with GPT-4o-mini
- Multi-image comparison: two images sent in a single API call
- Gemini Vision setup with dual Colab/local API key pattern
- `analyze_image_gemini()` helper using `Part.from_bytes()` — URL fetching + Gemini call in one function
- Gemini captioning and VQA on the same raccoon image — direct comparison with GPT-4o-mini output
- Provider comparison table: GPT-4o-mini vs Gemini 2.0 Flash on cost, context, and API format
- **Exercise:** Build a food-image recipe analyzer

> **Key Insight:** A multimodal model receives image patches as additional tokens — the attention mechanism sees visual tokens and text tokens in the same sequence. This is why the API call looks almost identical to a text-only call: you are still sending a `messages` list, just with an additional content type. The difference between GPT-4o-mini and Gemini is not capability — it is API format, context length (128K vs 1M tokens), and cost. Gemini's free tier makes it the natural choice for prototyping vision applications.

---

## The Conceptual Thread

The five notebooks tell a single story about the three layers of the LLM stack:

1. **The API layer (NB01).** Production LLM work rarely requires touching model weights directly. You send a prompt and receive a response via a standardized HTTP interface. The OpenAI API format is the industry standard — understanding it deeply, including how to route the same client to different backends, is the first practical skill.

2. **The model library layer (NB02).** Hugging Face gives you direct access to the weights, not just an API. The `pipeline()` abstraction hides the tokenization and logit decoding that Module 04 showed you explicitly. This layer is where you experiment with specialized models, evaluate alternatives, and prototype before deciding whether to call a cloud API or deploy locally.

3. **The architecture layer (NB03).** Behind every model is either an encoder (BERT) or a decoder (GPT). Encoders read everything at once and are optimized for understanding. Decoders read left-to-right and generate one token at a time. Modern chat LLMs are scaled-up decoders; embedding models are scaled-up encoders. Temperature and sampling parameters live at this layer.

4. **The application layer (NB04).** A working model is not a product. Gradio collapses the distance between a model and a shareable UI, making it possible to demo, test, and deploy without leaving Python.

5. **The multimodal extension (NB05).** The same architecture that processes text tokens can be extended to process image patches. The API call structure, the prompt engineering intuitions, and the provider trade-off reasoning all transfer directly from earlier notebooks.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 05 | Where it reappears |
|---|---|
| OpenAI `chat.completions.create` | Module 06 (prompt engineering), Module 08 (RAG query layer), Module 09 (optimization) |
| System / user message structure | Module 06 (prompt templates), Module 07 (fine-tuning data format) |
| Temperature and sampling | Module 06 (controlling output style), Module 09 (evaluation) |
| Hugging Face `pipeline()` and model IDs | Module 07 (fine-tuning base models), Module 08 (embedding models) |
| BERT encoder architecture | Module 08 (RAG: embedding models are encoders) |
| GPT decoder architecture | Module 07 (fine-tuning GPT-style models) |
| Model bias and safety | Module 09 (red-teaming and evaluation) |
| Gradio UI | Module 10 (Capstone: building deployable applications) |
| Multimodal vision-language models | Module 10 (Capstone: optional multimodal extensions) |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free GPU/CPU access, no local setup required)
- **Local:** Python 3.10+ with a virtual environment; each notebook installs its dependencies in the first code cell

### API Keys Required

| Notebook | Service | Where to get the key |
|---|---|---|
| NB01 | OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| NB01 (Gemini section) | Google AI Studio | [aistudio.google.com](https://aistudio.google.com) — free |
| NB02 | Hugging Face (optional) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — free, optional |
| NB05 | OpenAI | same as NB01 |
| NB05 (Gemini section) | Google AI Studio | same as NB01 Gemini |

Store keys as **Colab Secrets** (key icon in sidebar) or in a local `.env` file with keys named `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN`.

### Dependencies

| Library | Used in |
|---|---|
| `openai` | Notebooks 01, 05 |
| `ollama` | Notebook 01 |
| `google-genai` | Notebooks 01 (Gemini compat), 05 |
| `transformers` | Notebooks 02, 03 |
| `huggingface_hub` | Notebooks 02, 03 |
| `gradio` | Notebook 04 |
| `torch` | Notebooks 02, 03, 04 |
| `pillow` + `matplotlib` + `requests` | Notebook 05 |
| `python-dotenv` | All notebooks (local setup) |

### Run Order

Run notebooks in sequence: **NB01 → NB02 → NB03 → NB04 → NB05**. NB01 establishes the API mental model. NB02 gives you the model library. NB03 explains the architecture. NB04 shows deployment. NB05 extends to images.

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · LLM APIs: OpenAI, Ollama, and Gemini | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/01_openai_ollama.ipynb) |
| 02 · The Hugging Face Ecosystem | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/02_huggingface_tour.ipynb) |
| 03 · BERT vs GPT: Inside the Two Foundational Architectures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/03_bert_gpt.ipynb) |
| 04 · Building UIs with Gradio | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/04_gradio_ui.ipynb) |
| 05 · Multimodal AI: Vision-Language Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_05_LLMs/05_multimodal_models.ipynb) |
