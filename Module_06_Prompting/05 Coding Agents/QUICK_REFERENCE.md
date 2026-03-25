# Enhanced ReAct Agent — Quick Reference Guide

## What This Project Teaches

This is an **educational implementation** of how modern coding agents work internally:
- **Claude Code** (Cursor)
- **GitHub Copilot**
- **ChatGPT Code Interpreter**

By building it from scratch, students understand:
1. **Function calling**: How LLMs decide which tools to use
2. **Progressive disclosure**: Loading skills on demand to save tokens
3. **ReAct pattern**: Reason → Act → Observe → Loop
4. **Tool dispatch**: Routing LLM calls to actual Python functions
5. **Context management**: Why you can't fit everything in the system prompt

---

## Project Files

```
📁 05 Coding Agents/
├── README.md                    ← Start here (full guide)
├── enhanced_react_agent.ipynb   ← Main notebook (run this!)
├── sales.csv                    ← Sample data for Demo 2
└── skills/
    ├── map.md                   ← Skill catalog (master index)
    ├── code_helper.md           ← Skill: write & test Python code
    ├── data_analyst.md          ← Skill: analyze CSV/tabular data
    ├── debugger.md              ← Skill: read tracebacks & fix bugs
    ├── math_solver.md           ← Skill: solve equations & formulas
    ├── sql_analyst.md           ← Skill: write & run SQL queries
    └── compact.md               ← Context compaction instructions
```



---

## Key Concepts Explained

### **Skill Map (map.md)**
- Master index of available skills
- Agent reads this FIRST (always in system prompt)
- Small ~ 200 tokens, doesn't bloat context

### **Skills (code_helper.md, data_analyst.md)**
- Detailed instructions for specific task types
- Loaded ON DEMAND via `read_skill_file(skill_id)`
- Each skill knows which tools it can use
- Shows ReAct pattern example for that skill

### **Tools (run_python, read_file, etc.)**
- Functions the model can call
- Limited to what's needed (not 100+ tools)
- Agent sees the schema, not the implementation

### **ReAct Loop**
```
[Reflect] What do I know? What's needed?
    ↓
[Plan] Which tool should I use?
    ↓
[Act] Call the tool
    ↓
[Observe] What did I learn?
    ↓
Loop or [Respond] to user?
```

---

## For Instructors

### Use Case: Teaching GenAI
"Show students how Cursor/Copilot work without being a black box"

### Learning Journey
1. **Setup**: Install, run notebook
2. **Observe**: See agent load skills and call tools
3. **Trace**: Inspect history to see conversation flow
4. **Extend**: Add new skill or tool
5. **Deploy**: Wrap in API or Discord bot

### Discussion Questions
- Why load skills on demand instead of all at once?
- What breaks if you put all skills in the system prompt?
- How does the agent decide which skill to load?
- Why is the tool dispatcher pattern important?
- What happens when the context window fills up?

---

## What Was Fixed

✅ **Missing Imports**: Added `json`, `subprocess`  
✅ **Missing Tool**: Implemented `read_file()` and exposed it  
✅ **Error Handling**: Better exception handling in all tools  
✅ **Documentation**: Created README + VALIDATION_REPORT  
✅ **Skill Files**: Enhanced with real ReAct examples  
✅ **System Prompt**: Explicit ReAct format + constraints  
✅ **Demo Comments**: Each demo has educational annotations  

**See README.md for the full guide.**

---

## Running the Notebook

### Prerequisites
```bash
uv pip install openai python-dotenv
# or: pip install openai python-dotenv
```

### Setup API Key
**.env file**:
```
OPENAI_API_KEY=sk-your-key-here
```

Or **Google Colab**: Use Colab Secrets (see notebook comments)

### Run
```bash
jupyter notebook enhanced_react_agent.ipynb
```

Run cells in order:
1. ✅ Setup (imports)
2. ✅ Define tools
3. ✅ Build system prompt
4. ✅ Define agent loop
5. ▶️ Demo Task 1 (Code Helper)
6. ▶️ Demo Task 2 (Data Analyst)
7. ▶️ Demo Task 3 (Compaction)
8. 🔍 Inspect history

---

## Customization Ideas

### Add a New Skill
```markdown
# skills/web_searcher.md
## Purpose
Search the web for information

## Available Tools
- run_python() for using requests library
```

Add to `map.md`:
```markdown
| web_searcher | Search the web for current info | "Find latest Python news" |
```

Agent automatically discovers and uses it!

### Add a New Tool
1. Implement Python function: `def my_tool(args):`
2. Add to TOOLS list with OpenAI schema
3. Add to `dispatch_tool()` handler
4. Update skill files to mention it

### Change Model
```python
response = client.chat.completions.create(
    model="gpt-4-turbo",  # ← Change here
    ...
)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "OPENAI_API_KEY not found" | Add to .env file in notebook directory |
| Agent loops forever | Increase or remove step limit, or reduce max steps |
| "File not found" error | Skill files must be in `skills/` folder relative to notebook |
| Model doesn't load skill | Check skill filename matches skill_id (e.g., `code_helper.md` for `code_helper`) |
| Costs too high | Use smaller model (gpt-4o-mini is cheapest), set `verbose=False` |
| Python code fails | Agent sees stderr, might retry or ask for clarification |

---

## Educational Value

**This notebook is worth it because:**

1. **Concrete**: Not abstract - students see real code working
2. **Realistic**: Mirrors production agent architecture
3. **Extensible**: Students can add skills/tools easily
4. **Token-efficient**: Demonstrates cost optimization
5. **Understandable**: Simple enough to follow, complex enough to be real

---

## Resources

- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **ReAct Pattern**: https://arxiv.org/abs/2210.03629
- **Prompt Engineering**: https://platform.openai.com/docs/guides/prompt-engineering
- **Token Counting**: https://github.com/openai/tiktoken

---

**Ready to run! 🚀 Start with README.md for the full guide.**
