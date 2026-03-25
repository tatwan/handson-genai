# Skill Map

This is the **master catalog** of all available agent skills. Always read this first to decide which skill to load.

## Available Skills

| Skill ID | Description | Use When |
|---|---|---|
| **code_helper** | Write, debug, test, and explain Python code | "Write a function...", "Fix this code...", "How does... work?" |
| **data_analyst** | Analyze data files (CSV), summarize stats, answer questions | "Analyze sales.csv", "Which product...", "What's the average...?" |
| **debugger** | Read tracebacks, identify root causes, and verify fixes | "I'm getting a TypeError...", "This code crashes, can you fix it?" |
| **math_solver** | Solve equations, compute formulas, verify results step-by-step | "Solve 2x+5=13", "Calculate compound interest", "Is 1024 a power of 2?" |
| **sql_analyst** | Write and run SQL queries against a SQLite database | "Find the top 5 customers", "Join these tables", "How many orders in January?" |

## How the Agent Uses This Map

### Step 1: Read This Map
You (the agent) read this file to understand what skills exist.

### Step 2: Decide Which Skill to Load
Based on the user's request, pick the best skill.
- **User:** "Write a Python function"  
  → **Pick:** code_helper  
- **User:** "Analyze our sales data"  
  → **Pick:** data_analyst

### Step 3: Load the Full Skill
Call `read_skill_file(skill_id)` to load the detailed instructions.
- `read_skill_file("code_helper")` → loads code_helper.md with full instructions
- `read_skill_file("data_analyst")` → loads data_analyst.md with tools & examples

### Step 4: Use the Skill's Tools
Each skill has specific tools. From code_helper:
- `run_python(code)` - execute Python code
- `read_file(filepath)` - read source files

From data_analyst:
- `read_file(filepath)` - read CSV/data files
- `run_python(code)` - run pandas analysis code

### Step 5: Follow the ReAct Pattern
Every skill explains a ReAct loop:\
[Reflect] → [Plan] → [Tool Call] → [Observe] → [Respond]

## Progressive Disclosure Strategy

Why load skills on demand instead of including them all in the system prompt?

**Before (bad):**
```
System Prompt = Map + All Skills (~5,000 tokens)
Every request pays this cost
Limited space for conversation
```

**Now (good):**
```
System Prompt = Map only (~200 tokens)
Load code_helper when needed (~500 tokens)
Lots of space for actual conversation
Easy to add new skills without bloating the context
```

This is how **Claude Code**, **Cursor**, and **GitHub Copilot** work internally.

## When to Switch Skills
If the user's request changes topic, you might load a different skill.

Example conversation:
```
User: "Write me a prime checker function"
→ Load: code_helper

User: "Can you look at my sales.csv and find trends?"
→ Switch to: data_analyst

User: "I'm getting a TypeError on line 12, fix it"
→ Switch to: debugger

User: "What's the compound interest on $500 at 4% for 5 years?"
→ Switch to: math_solver

User: "Query my orders database for the top customers"
→ Switch to: sql_analyst

User: "Show me how to optimize that prime checker"
→ Back to: code_helper
```

## Compact Context
If the conversation gets long (>8 turns), call `compact_context()` to load compaction instructions and summarize.

---

**You are the agent. Always start by reading this map! 👇**
