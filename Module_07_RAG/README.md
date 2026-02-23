# Module 07: Retrieval-Augmented Generation (RAG)

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 2 of 3 | **Position:** Module 7 of 10 | **Notebooks:** 3

---

## Overview

Module 06 showed you how to prompt LLMs reliably and how to connect them to tools. This module answers a harder question: how do you ground an LLM in *your* specific knowledge — documents your model was never trained on?

The answer is **Retrieval-Augmented Generation (RAG)**: convert your documents into a searchable vector index, retrieve the most relevant passages at query time, and inject them into the prompt before the LLM generates an answer. The model never hallucinates facts it wasn't given — it only synthesizes what it retrieved.

The module builds from fundamentals to evaluation. NB01 constructs a full RAG pipeline over real research-paper PDFs using LangChain, adds conversation history awareness, and then demonstrates a second powerful pattern: a natural-language SQL agent over a relational database. NB02 shows the same retrieval pipeline implemented with LlamaIndex — fewer lines of code, same underlying concepts. NB03 adds the layer that matters most in production: systematic evaluation. It builds a golden-dataset benchmark, uses MLflow to track every experiment run, and applies LLM-as-a-Judge scoring (Faithfulness + Answer Relevance) to detect hallucinations and retrieval failures before users do.

---

## Where This Module Fits

```
Module 05: Large Language Models (LLMs)
      ↓
Module 06: Prompting and Agents
      ↓
Module 07: Retrieval-Augmented Generation (RAG)  ← you are here
      ↓
Module 08: Fine-Tuning
      ↓
Modules 09–10: Optimization · Capstone
```

Module 06 gave you agents that can call tools. RAG is a specific, crucial application of that idea: the retriever is a tool, your vector store is its data source, and the LLM synthesizes retrieved evidence into answers. This is the dominant architecture for production LLM applications. Fine-tuning (Module 08) changes what the model *knows*; RAG changes what information the model *sees* at inference time. Both are complementary, and Module 09's optimization work assumes you can evaluate RAG pipelines — the skill built in NB03.

---

## About the Dataset

This module uses **real ArXiv research papers** as the PDF corpus:

| Paper | Topic |
|-------|-------|
| Word2Vec (Mikolov et al., 2013) | Word embeddings, skip-gram, CBOW |
| GloVe (Pennington et al., 2014) | Global vector representations |
| BPE for NMT (Sennrich et al., 2015) | Byte Pair Encoding for neural machine translation |
| Word Embeddings Survey (Almeida & Xexéo, 2019) | Survey of embedding techniques |
| BPE Theory (Kozma & Voderholzer, 2024) | Theoretical analysis of optimal pair encoding |
| BERT (Devlin et al., 2018) | Bidirectional transformers for language understanding |
| E5 Embeddings (Wang et al., 2021) | Universal text embeddings |
| Additional NLP/ML papers | Various topics related to the course |

**Why PDFs instead of a benchmark dataset?** This is an intentional pedagogical choice:

- Students interact with content they've already studied in Modules 04–05 (embeddings, tokenization, transformers), so they can *verify* answers manually
- It mirrors a real-world use case: a company RAG-ing over its own internal documents (papers, manuals, reports)
- No external dataset download or Hugging Face credentials required — students open the notebooks and run immediately
- The papers are diverse enough that retrieval genuinely needs to discriminate (a question about BPE might pull from multiple papers; a hallucination would be obvious)

---

## Learning Objectives

By the end of this module you will be able to:

- Explain the Retrieve → Augment → Generate loop and why each step is necessary
- Load, split, embed, and store PDF documents in a Chroma vector store
- Build a history-aware retriever that reformulates follow-up questions into standalone queries
- Assemble a full conversational RAG chain using `create_retrieval_chain` and `create_stuff_documents_chain`
- Launch an interactive RAG chatbot with Gradio
- Build a natural-language SQL agent with `create_react_agent` (LangGraph) over a relational database
- Build the same RAG pipeline with LlamaIndex and compare the two frameworks' abstractions
- Persist and reload a LlamaIndex vector index to avoid re-embedding costs
- Design a golden evaluation dataset with diverse question types and ground-truth answers
- Use MLflow to track RAG experiment parameters, prompts, and artifacts reproducibly
- Apply LLM-as-a-Judge scoring (Faithfulness, Answer Relevance) using `mlflow.evaluate`
- Diagnose specific RAG failure patterns: hallucination, retrieval failure, out-of-scope queries
- Iteratively tune chunking and retrieval parameters using experiment comparison tables

---

## Module Notebooks

### 📓 01 · RAG with LangChain (PDF Chat + SQL Agent)

**File:** `01_rag_langchain.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_07_RAG/01_rag_langchain.ipynb)

**Concepts covered:**
- RAG pipeline anatomy: load → split → embed → store → retrieve → generate
- `PyPDFLoader` + `DirectoryLoader`: loading PDFs page by page with metadata (`source`, `page`, `total_pages`)
- `RecursiveCharacterTextSplitter`: hierarchical splitting (paragraph → sentence → word) with `chunk_size=1000`, `chunk_overlap=200`
- `OpenAIEmbeddings` (`text-embedding-3-small`): semantic vector representations
- `Chroma`: persistent vector store backed by local disk; L2-distance similarity search
- `create_history_aware_retriever`: reformulates follow-up questions into standalone queries using conversation history
- `create_stuff_documents_chain`: injects retrieved chunks into the prompt as `{context}`
- `create_retrieval_chain`: wires retriever + QA chain into a single invokable pipeline
- Gradio `ChatInterface`: zero-code web UI for interactive chat
- SQL agent with `create_react_agent` (LangGraph): natural-language queries over the Chinook SQLite database
- `SQLDatabaseToolkit`: bundles `list_tables`, `schema`, `query_checker`, and `query` tools for the agent
- Difference between `create_react_agent` (general, graph-based) and `create_sql_agent` (convenience helper with built-in SQL prompts)

**Lab builds:**
- 9-PDF corpus loaded (126 pages, 688 chunks after splitting)
- Chroma vector store created with `text-embedding-3-small` embeddings, persisted to `vector_db/`
- Similarity search demo: BPE query returning 5 ranked chunks with L2 scores
- Out-of-scope query demo: "What is the capital of France?" showing what retrieves when nothing is relevant
- History-aware multi-turn conversation: follow-up "repeat in bullet points" correctly answered
- Gradio chatbot UI (public share link via `share=True`)
- Chinook database download (SQLite, 11 tables: Album, Artist, Track, Customer, Employee, Invoice, etc.)
- `create_react_agent` SQL agent: "How many employees?" (COUNT query) and "Top 3 genres by tracks?" (JOIN + GROUP BY + ORDER BY)
- `create_sql_agent` comparison: verbose=True trace showing the full Thought → Action → Observation loop

> **Key Insight:** The history-aware retriever does not answer questions — it only *rewrites* them. The system prompt for this component says "do NOT answer, just reformulate." This separation of concerns (reformulation → retrieval → generation) is what makes multi-turn RAG reliable: each step has one job, and failures are easy to isolate.

---

### 📓 02 · RAG with LlamaIndex

**File:** `02_rag_llamaindex.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_07_RAG/02_rag_llamaindex.ipynb)

**Concepts covered:**
- LlamaIndex philosophy: data-centric framework optimized for indexing and retrieval vs. LangChain's general-purpose chain/agent focus
- `SimpleDirectoryReader`: automatic multi-format file loading (PDFs, text, markdown) with zero configuration
- `VectorStoreIndex.from_documents()`: one-call pipeline that chunks, embeds, and stores documents in an in-memory vector store
- `index.as_query_engine()`: single-turn question-answering engine over the index
- `response.source_nodes`: inspecting which documents (with similarity scores) contributed to an answer
- `index.storage_context.persist()`: serializing the index to disk to avoid re-embedding on restart
- `StorageContext.from_defaults()` + `load_index_from_storage()`: reloading a persisted index

**Lab builds:**
- Same 126-page PDF corpus loaded with `SimpleDirectoryReader("pdfs")`
- One-line index creation vs. NB01's 5-step setup (same result, less code)
- BPE query with source node inspection (scores: 0.889, 0.874 — higher is better in LlamaIndex's cosine similarity)
- Index persisted to `vector_db_llama/`
- Reload from disk and verify with the same BPE query
- Student challenge: build a RAG system over their own PDFs from scratch

> **Key Insight:** LlamaIndex and LangChain reach the same place — a retriever over your documents — via different abstractions. LlamaIndex hides chunking, embedding, and storage in a single call; LangChain exposes each step. Understanding NB01 first makes NB02 demystifying: LlamaIndex is a higher-level abstraction over the same primitives, not a different technology. When you need to customize chunk strategy, swap embedding models, or add metadata filters, you'll reach for LangChain's explicit pipeline. When you want something working in 10 lines, LlamaIndex wins.

---

### 📓 03 · RAG Evaluation and Observability with MLflow

**File:** `03_rag_evaluation.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_07_RAG/03_rag_evaluation.ipynb)

**Concepts covered:**
- Why "it looks good" is not enough: manual testing misses edge cases, silent regressions, and out-of-scope failures
- RAG failure taxonomy: hallucination (low faithfulness + high relevance), retrieval failure (high faithfulness + low relevance), complete failure (both low), phrase-sensitivity (inconsistent scores)
- LLM observability: traces (request → retrieval → generation), metrics (latency, tokens), logs (inputs/outputs)
- MLflow for LLM apps: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`, `mlflow.set_experiment()`, `mlflow.start_run()`
- `mlflow.langchain.autolog(log_traces=True)`: automatic trace capture for every LangChain call
- Golden dataset design: 8 questions covering factual lookup, theoretical complexity, comparative reasoning, benchmarks, and out-of-scope — all with ground-truth answers
- Batch inference loop: running every golden question through the RAG chain and capturing `answer` + `contexts`
- Quality guardrails before evaluation: empty answer detection, zero-context detection, average answer length
- `mlflow.evaluate()`: running the full evaluation dataset through LLM-as-a-Judge
- `faithfulness(model="openai:/gpt-4o-mini")`: does the answer contain only claims supported by the retrieved context? (hallucination detection, 1–5 scale)
- `answer_relevance(model="openai:/gpt-4o-mini")`: does the answer address the user's actual question? (1–5 scale)
- `mlflow.log_params()`, `mlflow.log_artifact()`, `mlflow.log_table()`: logging configurations, prompts, datasets for reproducibility
- Per-question failure analysis: isolating low-scoring questions to diagnose root cause
- Experiment comparison: querying all MLflow runs with `mlflow.search_runs()` to compare chunk sizes, retrieval k, and embedding models side by side
- MLflow UI: launching `mlflow ui --backend-store-uri sqlite:///mlflow.db` to view traces, metrics, and artifacts

**Lab builds:**
- Same RAG chain as NB01, now wrapped in MLflow autolog
- MLflow UI trace walkthrough: each retrieval + LLM call visible with timing and token counts
- 8-question golden dataset covering BPE, word embeddings, NNLMs, MTEB, and adversarial/out-of-scope questions
- Quality guardrails check before expensive LLM-Judge calls
- Full `mlflow.evaluate()` run with faithfulness + answer_relevance, logged to `sqlite:///mlflow.db`
- Per-question score table with failure pattern identification
- Experiment comparison table across multiple runs
- **Student Challenges:** (1) Tune `chunk_size` 1000 → 500 and compare faithfulness; (2) Add 3 new questions (factual, reasoning, out-of-scope); (3) Bonus: swap to `text-embedding-3-large` and compare

> **Key Insight:** Prompts and chunking parameters are *hyperparameters* of your RAG system, just like learning rate and batch size are hyperparameters of a neural network. Faithfulness and Answer Relevance are your loss functions. MLflow is your experiment tracker. The scientific method — baseline → hypothesis → experiment → compare → iterate — applies directly to LLM systems. A RAG pipeline without evaluation is not production-ready; it just hasn't failed visibly yet.

---

## The Conceptual Thread

The three notebooks tell a single story: how to build, understand, and trust a RAG system.

1. **LangChain RAG (NB01)** exposes every layer — loading, splitting, embedding, vector storage, retrieval, generation. You assemble each piece explicitly, which means you understand what can go wrong at each step. The SQL agent demonstrates that the same retrieval-augmentation principle applies to structured data: the agent retrieves schema information, generates SQL, executes it, and synthesizes the result — the same Retrieve → Augment → Generate loop, applied to databases.

2. **LlamaIndex RAG (NB02)** collapses the same pipeline into fewer calls. Seeing both frameworks back-to-back teaches that the abstractions differ but the concepts don't: a `VectorStoreIndex` is just a managed Chroma + embedding pipeline; `as_query_engine()` is just a retriever + LLM chain. Knowing NB01 means you can debug NB02 when it fails, and knowing NB02 means you can prototype faster.

3. **RAG Evaluation (NB03)** turns intuition into measurement. After building confidence in the system during NB01 and NB02, NB03 forces the question: *how good is it really?* The golden dataset, LLM-as-a-Judge metrics, and MLflow experiment tracking give you a rigorous answer — and a workflow for systematically improving it. This is the difference between a demo and a production system.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 07 | Where it reappears |
|---|---|
| Vector store + similarity search | Module 09 (optimization: ANN indexing, hybrid search) |
| Chunking strategy and overlap | Module 09 (chunk size as a tunable hyperparameter) |
| `create_retrieval_chain` + `create_stuff_documents_chain` | Module 10 (Capstone: document Q&A agents) |
| `create_react_agent` (SQL agent) | Module 10 (Capstone: multi-tool agents) |
| Golden dataset evaluation | Module 09 (evaluation methodology), Module 10 (Capstone: system evaluation) |
| MLflow experiment tracking | Module 09 (systematic hyperparameter search) |
| LLM-as-a-Judge (faithfulness, relevance) | Module 09 (evaluation framework), Module 10 (Capstone: quality gates) |
| RAG failure pattern analysis | Module 08 (fine-tuning as a complement to RAG for persistent failures) |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free CPU access, no local setup required)
- **Local:** Python 3.10+ with a virtual environment; each notebook installs its dependencies in the first code cell
- **Data files:** The `pdfs/` directory contains 9 ArXiv research papers — keep them in the same directory as the notebooks when running locally. No download required; they are included in the repository.

### API Keys Required

| Notebook | Service | Key name | Where to get it |
|---|---|---|---|
| NB01, 02, 03 | OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

All three notebooks use OpenAI for both embeddings (`text-embedding-3-small`) and the LLM (`gpt-4o-mini`). NB03 also uses `gpt-4o-mini` as the LLM-as-a-Judge for evaluation metrics.

Store keys as **Colab Secrets** (key icon in sidebar) or in a local `.env` file:
```
OPENAI_API_KEY=your-key-here
```

### Dependencies

| Library | Used in |
|---|---|
| `langchain` + `langchain-community` + `langchain-openai` | Notebooks 01, 03 |
| `langchain-chroma` | Notebooks 01, 03 |
| `langchain-text-splitters` | Notebooks 01, 03 |
| `langgraph` | Notebook 01 (SQL agent) |
| `llama-index` | Notebook 02 |
| `mlflow` | Notebook 03 |
| `pypdf` | Notebooks 01, 03 |
| `gradio` | Notebook 01 |
| `python-dotenv` | All notebooks (local setup) |

### Run Order

Run notebooks in sequence: **NB01 → NB02 → NB03**. NB01 builds the core RAG pipeline and conceptual foundation. NB02 provides the LlamaIndex perspective. NB03 applies systematic evaluation to the NB01 pipeline — understanding NB01's chain structure is necessary to interpret NB03's evaluation results.

> **Note on vector store reuse:** NB01 creates `vector_db/` on first run. NB03 loads the same `vector_db/` if it exists (to avoid re-embedding costs). Run NB01 before NB03 to populate it, or NB03 will create it from scratch.

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · RAG with LangChain | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_07_RAG/01_rag_langchain.ipynb) |
| 02 · RAG with LlamaIndex | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_07_RAG/02_rag_llamaindex.ipynb) |
| 03 · RAG Evaluation with MLflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_07_RAG/03_rag_evaluation.ipynb) |
