# Module 04: NLP — Understanding Language as Data

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 2 of 3 | **Position:** Module 4 of 10 | **Notebooks:** 3

---

## Overview

Every large language model you will use from Module 05 onward — GPT-4, Claude, Llama — is at its core an NLP system that learned from text. Before you can reason confidently about how these models work, you need to understand how computers represent language at all.

This module builds that understanding in three steps. First, you will trace the evolution from hand-split tokens and word-count vectors to dense word embeddings — covering the classical NLP stack that predates the Transformer era. Second, you will go deep on tokenization as a first-class engineering concern: how Byte Pair Encoding works, why it was invented, and how to use it to estimate costs and validate inputs before calling any LLM API. Third, you will work directly with production-grade semantic embeddings — the same `text-embedding-3-small` model available from OpenAI — to build a miniature semantic search engine and see exactly how meaning is encoded as geometry.

The progression is intentional: classical → algorithmic → production. Each step solves a concrete limitation of the previous one and previews a mechanism inside modern LLMs.

---

## Where This Module Fits

```
Module 01: ML Foundations
      ↓
Module 02: Deep Learning Primer
      ↓
Module 03: Overview of Generative AI
      ↓
Module 04: NLP — Understanding Language as Data  ← you are here
      ↓
Module 05: Large Language Models (LLMs)
      ↓
Modules 06–10: Prompting · Fine-Tuning · RAG · Optimization · Capstone
```

The conceptual bridge this module builds is critical. Module 03 showed you how neural networks can compress images into a latent space and generate from it. This module shows you that the same compression idea applies to language — that words, sentences, and entire documents can be mapped into vectors where geometry encodes meaning. Module 05 shows you what happens when you build a 70-billion-parameter model on top of exactly that idea.

---

## Learning Objectives

By the end of this module you will be able to:

- Tokenize text using multiple strategies (whitespace splitting, NLTK, spaCy) and explain when each is appropriate
- Implement Bag-of-Words from scratch and with scikit-learn, and demonstrate exactly why word order loss is a fundamental limitation
- Train a Word2Vec model on a custom corpus using gensim and use pre-trained GloVe embeddings to measure document similarity with cosine similarity
- Explain how Byte Pair Encoding (BPE) works step-by-step and why subword tokenization solves the unknown-word problem
- Use `tiktoken` to inspect token boundaries for any text, count tokens across different encodings, and understand why the same text yields different token counts
- Estimate LLM API costs before making a call, and validate whether an input fits within a model's context window
- Use OpenAI's `text-embedding-3-small` to embed text and implement a semantic search engine using cosine similarity
- Connect these NLP primitives — tokens, embeddings, cosine similarity — to how Transformer models and RAG pipelines work internally

---

## Module Notebooks

### 📓 01 · Introduction to NLP

**File:** `01_intro_to_nlp.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_04_NLP/01_intro_to_nlp.ipynb)

**Concepts covered:**
- Tokenization: naive space-splitting, NLTK `word_tokenize`, spaCy pipeline tokens
- Bag-of-Words (BoW): vocabulary construction, vector counting — built entirely from scratch
- BoW with scikit-learn `CountVectorizer` and the document-term matrix
- BoW's fundamental flaw: loss of word order (identical vectors for sentences with opposite meanings)
- Word2Vec: how training on word co-occurrence produces dense semantic vectors
- GloVe pre-trained embeddings (via gensim's model zoo): 50-dimensional vectors trained on Wikipedia + Gigaword
- Cosine similarity: measuring semantic closeness between document vectors
- Document retrieval: finding the most similar document to a query

**Lab builds:**
- Three-way tokenization comparison: raw split vs. NLTK vs. spaCy on the same contracting sentence
- BoW from scratch: vocabulary → integer index → count matrix → pandas DataFrame
- BoW limitation demo: "The dog bit the man" and "The man bit the dog" producing identical vectors
- Word2Vec training on a small AI/ML corpus with gensim; document vectors via mean pooling
- Pre-trained GloVe embeddings for three-document retrieval: space, ocean, and technology documents; query: "astronaut travels to the moon"
- **Assignment:** Train Word2Vec on a larger corpus from Project Gutenberg; implement word analogies (`king − man + woman`); build a full similarity search with top-5 ranked results

> **Key Insight:** Bag-of-Words collapses text into a count over a fixed vocabulary — fast and interpretable, but it loses all word order and treats "dog bit man" identically to "man bit dog." Word embeddings solve this by learning from *context*: words that appear in similar contexts end up close in vector space. The jump from BoW to embeddings is the jump from counting to understanding.

---

### 📓 02 · Tokenization: From Fundamentals to Tiktoken

**File:** `02_tokenization.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_04_NLP/02_tokenization.ipynb)

**Concepts covered:**
- Regex-based tokenization and building encode/decode functions from a vocabulary dictionary
- The unknown-word (OOV) problem with word-level tokenization and why it makes scaling impossible
- Byte Pair Encoding (BPE): iterative pair merging from character-level to subword tokens
- Why BPE eliminates OOV: any word decomposes into known character sequences
- Tiktoken: OpenAI's production BPE library — encodings `cl100k_base` (GPT-4/3.5), `p50k_base`, `o200k_base` (GPT-4o)
- Visualizing token boundaries: why "hello_world" is one token but "hello world" is two, why emojis are 3 tokens
- Token-based pricing: input tokens, output tokens, cost-per-million rates by model
- Context window limits: max tokens per model and why both input and output count against the limit
- Dynamic text truncation: fit any input within a model's context window programmatically

**Lab builds:**
- Vocabulary encoder/decoder from scratch demonstrating the OOV failure mode
- BPE step-by-step: three manual merge iterations on `{"low", "lower", "newest", "widest"}` with pair frequency counts visible at each step
- Tiktoken encoding comparison: `cl100k_base` vs `p50k_base` vs `r50k_base` on the same sentence
- Token visualizer: token ID, decoded string, and boundary for natural language, numbers, code, and emojis
- Cost estimator: functions for single queries, document summarization, cross-model comparison, and batch processing (5 → 10,000 reviews)
- Input validator: per-model context limit check with utilization percentage and remaining capacity
- Text truncator: automatically trim input to fit any model's context window with safety margin

> **Key Insight:** BPE doesn't tokenize — it *learns* to tokenize. It finds the compression that best matches the corpus's statistical structure. This is why GPT-4 can handle rare words and code without an explicit dictionary: any input decomposes into known subword pieces. Knowing this lets you predict where tokenization surprises will appear (special characters, numbers, rare domain terms) and design your prompts accordingly.

---

### 📓 03 · Embeddings and Semantic Similarity

**File:** `03_embeddings.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_04_NLP/03_embeddings.ipynb)

**Concepts covered:**
- Tiktoken tokenization review: inspecting subword boundaries and token IDs for GPT-4 encoding
- OpenAI `text-embedding-3-small`: a 1,536-dimensional dense embedding model trained to capture semantic meaning
- How embeddings represent meaning as geometry: related concepts are nearby; unrelated concepts are distant
- Cosine similarity formula: dot product normalized by vector magnitudes, yielding values in [−1, 1]
- Semantic search: embedding a query, computing similarity against a document corpus, ranking results

**Lab builds:**
- `show_tokens()`: decodes and prints every token ID for any input text — demonstration with emojis and multilingual text
- `get_embedding()`: wraps the OpenAI embeddings API and returns a numpy vector
- Word-level comparisons: Apple vs. Banana (0.39), Apple vs. iPhone (0.70), Banana vs. SpaceX (0.18) — geometry reflects intuition
- Sentence-level comparisons: "The cat is happy" vs. "The kitten is joyful" (0.80) vs. "The stock market crashed" (0.02)
- Mini semantic search engine: 7-document corpus, free-text query, results ranked by cosine similarity
- **Student Activity 1 — Tokenizer Playground:** experiment with emojis, non-Latin scripts, and invented words
- **Student Activity 2 — Mini Semantic Search:** extend the corpus and try custom queries

> **Key Insight:** The embedding of "The kitten is joyful" is close to "The cat is happy" not because they share words (they share only "the" and "is") but because the model learned from billions of examples that *cats*, *kittens*, *happiness*, and *joy* appear in similar contexts. This is precisely the mechanism that makes semantic search, RAG retrieval, and LLM comprehension possible — and why you cannot achieve these results with BoW or keyword matching.

---

## The Conceptual Thread

The three notebooks tell a single story of progressively more powerful representations:

1. **BoW counts words.** It is interpretable, fast, and completely blind to meaning. "Dog bit man" and "Man bit dog" are the same vector. This is not fixable by counting more carefully — the information was thrown away at the design level.

2. **Word2Vec and GloVe learn meaning from context.** Instead of counting, they train a neural network to predict surrounding words. Words that appear in similar contexts (bank, finance, loan) cluster in the vector space. Cosine similarity over these vectors gives you a semantic distance — not a keyword-overlap score.

3. **BPE tokenizes without dictionaries.** It compresses vocabulary by finding statistical regularities in character sequences. This is why you can tokenize any word, in any language, without an explicit list of valid tokens — and why modern LLMs can handle domain-specific jargon, code, URLs, and rare names gracefully.

4. **Production embeddings encode full sentences and documents.** `text-embedding-3-small` goes beyond word-level similarity: it encodes the *meaning of the whole text* into a single 1,536-dimensional vector, trained with contrastive learning on pairs of semantically related texts. This is the technology that powers semantic search, RAG retrieval, and document classification in production systems.

Each step solves the fundamental limitation of the previous one. And the endpoint — dense semantic embeddings + cosine similarity — is exactly the retrieval mechanism at the core of every RAG pipeline you will build in Module 08.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 04 | Where it reappears |
|---|---|
| Tokenization (splitting text into subword units) | Module 05 (LLMs: vocabulary, input IDs, attention over token sequences) |
| BPE and tiktoken | Module 05 (GPT tokenizer internals), Module 06 (prompt token counting and optimization) |
| Cosine similarity as a distance measure | Module 08 (RAG: vector database retrieval uses cosine or dot product similarity) |
| Dense embeddings (vectors that encode meaning) | Module 08 (document chunks → embeddings → Chroma vector store → retrieval) |
| Context window limits and token counting | Module 06 (prompt engineering), Module 07 (fine-tuning with sequence length constraints) |
| Word2Vec/GloVe as representation learning | Module 05 (Transformers replace static word vectors with contextual representations from attention) |
| Document vectorization and similarity search | Module 08 (RAG pipeline: same idea, production-scale with Chroma and LangChain) |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free GPU/CPU access, no local setup required)
- **Local:** Python 3.10+ with a virtual environment; each notebook installs dependencies in its first cell automatically
- **NB03 note:** Requires an OpenAI API key for the embeddings API. In Colab, store the key as a Secret named `OPENAI_API_KEY`. Locally, set it in a `.env` file and uncomment the `dotenv` block in the first cell.

### Dependencies

| Library | Used in |
|---|---|
| `nltk` | Notebook 01 |
| `spacy` + `en_core_web_sm` | Notebook 01 |
| `gensim` | Notebook 01 |
| `scikit-learn` | Notebook 01 |
| `tiktoken` | Notebooks 02, 03 |
| `openai` | Notebook 03 |
| `numpy` + `pandas` + `matplotlib` | All notebooks |

### Run Order

Run notebooks in sequence: **NB01 → NB02 → NB03**. NB01 establishes the problem (why BoW fails, why embeddings help). NB02 explains the mechanism behind LLM tokenization (BPE) and teaches production-essential token counting. NB03 uses production embeddings to bring the conceptual arc to a modern close.

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · Introduction to NLP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_04_NLP/01_intro_to_nlp.ipynb) |
| 02 · Tokenization: From Fundamentals to Tiktoken | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_04_NLP/02_tokenization.ipynb) |
| 03 · Embeddings and Semantic Similarity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_04_NLP/03_embeddings.ipynb) |
