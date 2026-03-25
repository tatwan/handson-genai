# Skill: Debugger

## Purpose
Read Python tracebacks, identify root causes, and propose + test a fix. Teaches iterative error recovery — the core of any real agent's resilience.

## When to Use This Skill
- "I'm getting a TypeError / AttributeError / KeyError..."
- "This code is broken, can you fix it?"
- "Why does my function return the wrong answer?"
- "Fix this bug: [paste traceback]"

## Available Tools
| Tool | Purpose |
|------|---------|
| `read_file(filepath)` | Read the buggy source file |
| `run_python(code)` | Reproduce the error, then test the fix |

## ReAct Example: "Fix this — TypeError: unsupported operand type(s) for +: 'int' and 'str'"

```
[Reflect] The error is a TypeError on a + operation.
  - One operand is int, the other is str
  - Common cause: reading from CSV/input gives strings by default
  - I need to see the code and reproduce the error first

[Plan]
  1. Reproduce the error with run_python() (confirm I understand it)
  2. Identify the exact line
  3. Apply the fix (cast to correct type)
  4. Re-run to confirm it's resolved

[Tool Call #1] run_python("
# Reproduce bug
data = {'price': '25', 'quantity': 3}   # price is str (simulating CSV read)
total = data['price'] * data['quantity']  # BUG: str * int not the same as int * int
print(total)
")

[Observe] Output:
  252525   # repeated string, not 75 — wrong result, not even an error here
  # The real bug is the logic using str multiplication

[Tool Call #2] run_python("
# Fix: cast price to int before multiplying
data = {'price': '25', 'quantity': 3}
total = int(data['price']) * data['quantity']
print(f'Total: {total}')   # should print 75
")

[Observe] Output:
  Total: 75  ✓

[Respond] The bug was a type mismatch — 'price' was read as a string (common
when loading CSV data). Fix: wrap it with int() or float() before arithmetic.
Here's the corrected line:
  total = int(data['price']) * data['quantity']
```

## Best Practices
- **Always reproduce the error first** with `run_python()` — never guess
- Read the full traceback: the LAST line is the error, the line above it is where it happened
- Test the fix before reporting it — agent credibility depends on verified answers
- Explain WHY the bug existed, not just what to change
- If the fix introduces a new error, keep iterating (the loop handles this!)

## Common Bug Patterns to Know

| Error | Likely Cause | Fix Pattern |
|-------|-------------|-------------|
| `TypeError: ... 'str'` | CSV/input gives strings | `int()`, `float()` cast |
| `KeyError: 'x'` | Dict key missing | Check keys first, use `.get()` |
| `IndexError` | List access out of range | Check `len()` before indexing |
| `AttributeError: 'NoneType'` | Function returned None | Check return value |
| `IndentationError` | Mixed tabs/spaces | Use spaces only |
