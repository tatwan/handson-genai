# Enhanced ReAct Agent — Educational Implementation

This folder demonstrates **how modern coding agents work** by implementing a simplified but realistic agent system from scratch.

## Learning Objectives

By working through this code, students will understand:

1. **Progressive Disclosure**: Only load skill instructions when needed (keeps context efficient)
2. **ReAct Loop**: Reason → Plan → Act → Observe → Repeat
3. **Function Calling**: How models decide which tools to use
4. **Tool Dispatch**: Mapping tool calls to actual Python functions
5. **Context Management**: How to compact long conversations

This mirrors how **Claude Code**, **Cursor**, **GitHub Copilot**, and **ChatGPT** work internally.

---

## Project Structure

```
05 Coding Agents/
├── README.md                          ← You are here
├── enhanced_react_agent.ipynb         ← Main notebook (start here!)
├── sales.csv                          ← Sample data for Demo 2
└── skills/
    ├── map.md                         ← Master catalog of skills
    ├── code_helper.md                 ← Skill: write & test Python code
    ├── data_analyst.md                ← Skill: analyze CSV/tabular data
    ├── debugger.md                    ← Skill: read tracebacks & fix bugs
    ├── math_solver.md                 ← Skill: solve equations & formulas
    ├── sql_analyst.md                 ← Skill: write & run SQL queries
    └── compact.md                     ← Instructions for context compaction
```

---

## Getting Started

### Prerequisites

```bash
# Install dependencies
uv pip install openai python-dotenv

# Or with pip:
pip install openai python-dotenv
```

### API Key Setup

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=sk-...your-key...
```

Or in Google Colab, add your API key in Colab Secrets (see comments in notebook).

### Run the Notebook

```bash
jupyter notebook enhanced_react_agent.ipynb
```

---

## How It Works

### The Architecture

```
User Request
    ↓
[System Prompt = map.md + rules]
    ↓
[Model sees user message]
    ↓
[Model decides: "Which skill should I load?"]
    ↓
[Call read_skill_file("skill_id")]  ← Progressive disclosure!
    ↓
[Skill file loaded into history]
    ↓
[Model calls tools: read_file(), run_python()]
    ↓
[Tools return results]
    ↓
[Loop until model has final answer]
```

### Key Concepts

| Concept | In This Code | In Real Agents |
|---------|-------------|-----------------|
| Skill Map | `map.md` | Copilot's `SKILL.md` registry |
| Progressive Loading | `read_skill_file()` | Load modules on demand, not all at start |
| Tool Calling | `run_python()`, `read_file()` | Code execution, file ops, API calls |
| Tool Dispatch | `dispatch_tool()` function | Route LLM calls to actual functions |
| Context Compaction | `compact_context()` | Summarize long conversations |
| ReAct Pattern | Agent loop in `run_agent()` | Every capable agent uses this |

---

## Demo Tasks

The notebook includes three demo tasks:

### Task 1: Code Helper
```python
# Agent reads map.md → sees "code_helper" is available
# → loads code_helper.md
# → uses run_python() to write AND test code

run_agent("Write a function that checks if a number is prime")
```

**What you'll see:**
- Agent thinks step-by-step (ReAct format)
- Agent calls `read_skill_file("code_helper")`
- Agent calls `run_python()` with the code
- Agent explains the result

### Task 2: Data Analyst (Skill Switching)
```python
# Agent switches from "code_helper" to "data_analyst"
# because the request is about CSV data

run_agent("Which product has the highest revenue?")
```

**What you'll see:**
- Agent picks a different skill
- Agent reads the CSV file
- Agent uses pandas to analyze data
- Demonstrates adaptive behavior

### Task 3: Context Compaction
```python
# After several turns, agent compacts history
# This frees up tokens without losing context

run_agent("Summarize what we've done so far")
```

**What you'll see:**
- Agent structures summary as per compact.md
- History is conceptually "reset" with summary

---

## Customization Ideas

### Add a New Skill

1. Create `skills/math_tutor.md`:
   ```markdown
   # Skill: Math Tutor
   
   ## Purpose
   Help students solve math problems step-by-step
   
   ## Available Tools
   - run_python(code) for numerical computation
   ```

2. Add to `skills/map.md`:
   ```markdown
   | math_tutor | Solve math problems step-by-step | "Solve 2x + 5 = 13" |
   ```

3. Agent will automatically discover and use it!

### Modify Tool Behavior

Edit the tool implementations in the notebook:
- `read_file()` — add caching or filtering
- `run_python()` — add output sanitization for safety
- Add new tools: `run_bash()`, `fetch_url()`, etc.

### Change the Model

Replace `"gpt-4o-mini"` with any OpenAI model:
```python
response = client.chat.completions.create(
    model="gpt-4-turbo",  # <- Change here
    ...
)
```

---

## Educational Connections

This notebook teaches concepts from Module 06: Prompting:

- **Function Calling** (Session 02): How models choose tools → skill map decides!
- **Prompt Engineering** (Session 01): System prompt rules encoded in behavior
- **ReAct Agents** (Session 04): The core loop demonstrated here
- **Tool Design** (Implicit): Each skill defines its own tools

---

## Important Notes for Students

1. **Always run Setup cells first**: Imports and client setup are required
2. **API costs**: Each agent query costs tokens. Use `verbose=False` for cheaper iterations
3. **Tool limits**: Only 4 tools are exposed (as designed). Real agents have 100+
4. **Token efficiency**: Try running with just the map, then with different loads
5. **Error handling**: Model might retry or ask for clarification (realistic!)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "OPENAI_API_KEY not found" | Add to .env or Colab Secrets |
| Agent loops forever | Increase safety limit (step > 10) |
| Model doesn't load skill | Check skill filename matches skill_id |
| Python code fails | Agent sees stderr, but might retry |
| Costs too high | Use gpt-4o-mini (cheapest), reduce verbose=True runs |

---

## Next Steps

1. **Extend**: Add more skills (math, web search, database)
2. **Optimize**: Try different prompting strategies for each skill
3. **Integrate**: Use this as foundation for your own project
4. **Deploy**: Wrap this in a FastAPI server or Discord bot

---

**Happy learning! 🎉**
