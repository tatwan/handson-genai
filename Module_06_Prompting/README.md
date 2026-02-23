# Module 06: Prompting and Agents

**Course:** Hands-On Generative Artificial Intelligence
**Day:** 2 of 3 | **Position:** Module 6 of 10 | **Notebooks:** 4

---

## Overview

Module 05 showed you how LLMs work and how to call them. This module shows you how to make them do exactly what you want — and how to extend them beyond their training data by connecting them to external tools.

The module has two distinct arcs. The first (NB01) is about the craft of prompting: how to structure inputs so that the same model that gives you a vague or wrong answer becomes reliable, precise, and expert-level. Every technique builds on the previous — from zero-shot to few-shot to chain-of-thought to self-consistency to tree of thoughts to multi-step prompt chains. The second arc (NB02–NB04) is about tool use and agents: giving LLMs the ability to call real functions, query databases, fetch live data, and reason over multi-step tool loops. NB04 culminates in building a ReAct agent from scratch — the same Thought/Action/Observation loop that underlies modern agentic frameworks.

By the end of this module you understand not just *how* to prompt, but *why* certain prompting strategies work on a mechanistic level, and you have built the full tool-use loop — first manually, then with LangChain — and implemented a working ReAct agent.

---

## Where This Module Fits

```
Module 05: Large Language Models (LLMs)
      ↓
Module 06: Prompting and Agents  ← you are here
      ↓
Module 07: Fine-Tuning
      ↓
Module 08: Retrieval-Augmented Generation (RAG)
      ↓
Modules 09–10: Optimization · Capstone
```

Module 05 gave you the model layer. This module gives you the interaction layer on top of it. Prompt engineering is how you get reliable behavior from a stochastic model; tool use is how you ground that model in real-world data. Both skills carry directly into every subsequent module: RAG is prompt engineering + tool use applied to document retrieval; fine-tuning changes what the model knows; the capstone combines everything.

---

## Learning Objectives

By the end of this module you will be able to:

- Write prompts with clear role, context, task, format, and example components and explain why each part matters
- Apply zero-shot, few-shot, and chain-of-thought prompting and choose between them based on task complexity
- Implement self-consistency (majority vote over multiple reasoning paths) to improve reliability on complex problems
- Use Tree of Thoughts to explore and evaluate multiple solution approaches before committing
- Design multi-step prompt chains where each step's output feeds the next, with intermediate validation
- Define OpenAI function calling tools with JSON schema and manage the full four-step lifecycle (initial call → tool selection → execution → result submission)
- Use the LangChain `@tool` decorator to create tools with automatic schema generation, and build an agent with `create_tool_calling_agent` and `AgentExecutor`
- Build a ReAct agent from scratch using the Thought/Action/PAUSE/Observation loop
- Identify and fix common prompting mistakes (vagueness, missing format, contradictory instructions)

---

## Module Notebooks

### 📓 01 · Prompting Techniques

**File:** `01_prompting_techniques.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/01_prompting_techniques.ipynb)

**Concepts covered:**
- Anatomy of a prompt: role/persona, context, task/instruction, format, examples
- Zero-shot prompting: when pre-trained knowledge is sufficient; standard classification, summarization, code generation
- Few-shot prompting: teaching the model custom output formats through examples; domain-specific conventions
- Chain-of-Thought (CoT): `"Think step by step"` and few-shot CoT with explicit reasoning traces
- Self-consistency: generating N reasoning paths at temperature > 0, extracting answers with regex, taking the majority vote
- Tree of Thoughts (ToT): three-stage explore-evaluate-develop loop; ToT + expert persona
- Structured output prompting: JSON extraction, markdown table generation
- Role/persona prompting: same question answered at three expertise levels (teacher / architect / CTO)
- Advanced techniques: self-refinement (draft → critique → improve), constraint-based prompting (MUST / MUST NOT)
- Prompt chaining: a four-step document processing pipeline (extract → analyze → summarize → format), chaining with JSON validation

**Lab builds:**
- `chat()` helper wrapping **Gemini 2.0 Flash** (primary, free tier) via OpenAI compatibility — or `gpt-4o-mini` as a commented backup; all downstream cells work unchanged
- Bad-vs-good prompt comparison showing the impact of specificity and format
- Zero-shot: sentiment classification, two-sentence summarization, function generation
- Few-shot: custom sentiment labeling with reasons; structured job-posting data extraction
- CoT: apple discount word problem with and without reasoning; few-shot CoT on geometry
- Self-consistency: 5-path majority vote on a classic "all but N die" problem with regex answer extraction
- Tree of Thoughts: city traffic problem — brainstorm 3 approaches → score on 4 criteria → develop winner
- ToT + expert persona: system architecture for 10M-user notification system
- Self-refinement: email declining a meeting invitation, with explicit critique step
- Four-step earnings-report pipeline: metric extraction → analysis → investor summary → formatted report card
- Prompt chain with validation: contact extraction with JSON parse + email format check
- **Exercises:** classification system design, CoT code review, structured entity extraction

> **Key Insight:** Prompting is not a collection of tricks — it is a systematic approach to communicating intent to a probabilistic system. Chain-of-thought works because it forces the model to allocate computation to intermediate steps rather than jumping to an answer. Self-consistency works because errors in reasoning are rarely correlated across independent paths. Tree of Thoughts works because the best answer to a complex problem is rarely on the first branch explored.

---

### 📓 02 · Function Calling (Manual)

**File:** `02_function_calling.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/02_function_calling.ipynb)

**Concepts covered:**
- Function calling (OpenAI Tools API): how the model selects and parameterizes tools without executing them
- The four-step tool lifecycle: (1) send prompt + tool schemas → (2) check for tool calls → (3) execute tool locally → (4) submit result with `role="tool"` and get final answer
- JSON schema for tool definitions: `type`, `properties`, `description`, `required`
- Multi-tool calls: a single prompt can trigger multiple parallel tool calls
- DuckDB for in-notebook SQL queries: loading CSV data and querying with parameterized statements
- Open-Meteo API: live weather data without an API key (free, open)

**Lab builds:**
- DuckDB setup: `flight_data.csv` and `fun_facts.csv` loaded into `city_tour.db`
- Three tool implementations: `get_weather()` (Open-Meteo REST API), `get_flight()` (DuckDB query), `get_fact()` (DuckDB query) — all with parameterized SQL and JSON returns
- Full JSON tool schemas for all three tools
- `run_conversation()` orchestrator implementing the four-step lifecycle manually
- Four demos: weather by coordinates, Paris→London flight lookup, Tokyo fun fact, multi-tool New York→London + London fact

> **Key Insight:** The model never executes your code. It reads the tool schemas, decides which tool(s) to call and with what arguments, and returns a structured request. Your code catches that request, runs the actual function, and returns the result. The model then synthesizes the data into a natural language answer. Understanding this separation — the model reasons, your code acts — is the foundation for building any production AI system that interacts with external data.

---

### 📓 03 · Function Calling with LangChain

**File:** `03_function_calling_langchain.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/03_function_calling_langchain.ipynb)

**Concepts covered:**
- LangChain `@tool` decorator: automatic JSON schema generation from Python type hints and docstrings — no manual schema writing
- `ChatOpenAI` and `ChatPromptTemplate` from LangChain's OpenAI integration
- `create_tool_calling_agent`: modern LangChain agent that uses the OpenAI tool-calling API
- `AgentExecutor`: the automated loop that replaces the manual `run_conversation()` from NB02
- `agent_scratchpad` placeholder: how LangChain tracks the tool call/result history inside the prompt
- `verbose=True`: observing the full agent reasoning chain in the output

**Lab builds:**
- Same three tools as NB02 — rewritten with `@tool` decorator; type hints drive schema validation
- `ChatOpenAI(model="gpt-4o-mini", temperature=0)` LLM setup
- Prompt template with system message + human message + `{agent_scratchpad}` placeholder
- Agent + executor assembly with `verbose=True`
- Three demos with visible chain output: weather query, Paris→London flight, complex multi-tool New York→London + London fact
- Side-by-side comparison: LangChain's auto-generated schema vs. the manual JSON from NB02

> **Key Insight:** LangChain's `@tool` decorator does exactly what you did manually in NB02 — it reads your function's type hints and docstring and generates the JSON schema. `AgentExecutor` runs the same four-step loop you wrote by hand. Understanding NB02 first makes NB03 demystifying: LangChain is automation over the same primitives, not magic. This also explains why debugging LangChain agents requires understanding the underlying tool-calling protocol.

---

### 📓 04 · ReAct Agent from Scratch

**File:** `04_react_agent.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/04_react_agent.ipynb)

**Concepts covered:**
- The ReAct framework (Reasoning + Acting): interleaving Thought, Action, PAUSE, and Observation in a structured loop
- Benefits over standard CoT: reduced hallucination (real data, not memorized), adaptability (replan on new observations), explainability (visible thought traces), access to live external data
- The `Agent` class: conversation history management, `__call__` for stateful interactions, `execute()` for LLM calls
- ReAct prompt structure: defining the loop format + available actions + a worked example in the system prompt
- Action parsing: using `re.compile()` to extract tool name and arguments from LLM output
- The `query()` loop: max-turn safety limit, action dispatch, observation injection, termination detection

**Lab builds:**
- `Agent` class with message history, callable interface, and `gpt-4o-mini` execution
- Simple helpful-assistant demonstration of conversation history accumulation
- Two tools: `calculate()` (Python arithmetic via `eval()`, with security note) and `average_dog_weight()` (mock database lookup)
- ReAct system prompt defining the Thought/Action/PAUSE/Observation/Answer format
- `action_re` regex for parsing `Action: tool_name: argument` lines
- `query()` orchestration loop with max-turn protection
- Full multi-step demo: "combined weight of border collie + scottish terrier" requiring 3 tool calls (2 lookups + 1 calculation) — showing the complete Thought→Action→PAUSE→Observation→Answer trace

> **Key Insight:** The ReAct agent is just a conversation with a carefully designed system prompt that teaches the model a specific output format (Thought/Action/PAUSE), combined with a Python loop that parses that format and calls real functions. There is no magic — it is prompt engineering + the tool-calling primitives from NB02 combined into a loop. Every modern agentic framework (LangChain agents, LlamaIndex agents, AutoGen, CrewAI) is a more sophisticated version of exactly this pattern.

---

## The Conceptual Thread

The four notebooks tell a single story that ends with a working autonomous agent:

1. **Prompt engineering (NB01)** teaches the model to behave reliably without changing it. Zero-shot, few-shot, and CoT are all techniques for shaping the model's internal reasoning process through natural language. Self-consistency and ToT extend this to multi-path exploration. Prompt chaining structures complex workflows as a sequence of reliable single-purpose calls. These skills underpin everything — every agent, every RAG query, every fine-tuning data format starts with a well-engineered prompt.

2. **Function calling — manual (NB02)** gives the model hands. The four-step lifecycle (send → receive tool request → execute locally → submit result) is the foundation of all agentic behavior. Implementing it manually demystifies every framework that builds on top of it.

3. **Function calling — LangChain (NB03)** shows what happens when you automate the manual loop. The `@tool` decorator removes schema boilerplate; `AgentExecutor` removes the loop. The conceptual content is identical to NB02; the engineering overhead is dramatically lower. Knowing both levels makes you a more effective debugger.

4. **ReAct agent (NB04)** combines everything: a `chat()` helper (NB01 pattern), tool definitions (NB02 pattern), and a reasoning loop (the Thought/Action/Observation format). The key addition is *transparency* — the model's reasoning is visible at every step, which is both what makes ReAct explainable and what makes it debuggable.

---

## How This Module Connects to the Rest of the Course

| Concept from Module 06 | Where it reappears |
|---|---|
| Few-shot and structured output prompting | Module 07 (fine-tuning data formatting), Module 09 (evaluation prompt templates) |
| Chain-of-thought and reasoning patterns | Module 09 (evaluation with reasoning), Module 10 (Capstone: complex agent tasks) |
| Prompt chaining with validation | Module 08 (RAG: retrieval → augmentation → generation pipeline) |
| Function calling (tool definitions + lifecycle) | Module 08 (RAG tools), Module 10 (Capstone agent tools) |
| LangChain `@tool` and `AgentExecutor` | Module 08 (LangChain RAG chains), Module 10 (Capstone agent framework) |
| ReAct Thought/Action/Observation loop | Module 10 (Capstone: multi-step autonomous agents) |
| DuckDB for structured data queries | Module 08 (structured data retrieval in RAG pipelines) |

---

## Getting Started

### Environment

- **Recommended:** Google Colab (free CPU access, no local setup required)
- **Local:** Python 3.10+ with a virtual environment; each notebook installs its dependencies in the first code cell
- **Data files:** `flight_data.csv` and `fun_facts.csv` must be in the same directory as NB02 and NB03 when running locally

### API Keys Required

| Notebook | Service | Key name | Where to get it |
|---|---|---|---|
| NB01 | Google AI Studio (primary) | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — **free** |
| NB01 | OpenAI (backup) | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| NB02, 03, 04 | OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

NB01 uses **Gemini 2.0 Flash by default** (free tier). To switch to OpenAI, comment out the Gemini block in the setup cell and uncomment the two OpenAI lines — no other cell changes are needed.

Store keys as **Colab Secrets** (key icon in sidebar) or in a local `.env` file.

### Dependencies

| Library | Used in |
|---|---|
| `openai` | All notebooks |
| `langchain` + `langchain-openai` + `langchain-community` | Notebook 03 |
| `duckdb` | Notebooks 02, 03 |
| `requests` | Notebooks 02, 03 |
| `python-dotenv` | All notebooks (local setup) |

### Run Order

Run notebooks in sequence: **NB01 → NB02 → NB03 → NB04**. NB01 builds the prompting foundation. NB02 and NB03 teach tool use at two levels of abstraction. NB04 synthesizes everything into a working agent.

### Quick Launch

| Notebook | Open in Colab |
|---|---|
| 01 · Prompting Techniques | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/01_prompting_techniques.ipynb) |
| 02 · Function Calling (Manual) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/02_function_calling.ipynb) |
| 03 · Function Calling with LangChain | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/03_function_calling_langchain.ipynb) |
| 04 · ReAct Agent from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tatwan/handson-genai/blob/main/Module_06_Prompting/04_react_agent.ipynb) |
