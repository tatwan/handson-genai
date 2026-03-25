# Skill: Code Helper

## Purpose
Write, debug, test, and explain Python code. Teachers and learners can request functions, fixes, or explanations.

## When to Use This Skill
- "Write a function that..."
- "Fix this bug..."
- "Explain how... works"
- "Generate code that..."

## Available Tools
| Tool | Purpose |
|------|---------|
| `run_python(code)` | Execute Python code and see the output |
| `read_file(filepath)` | Read source files to understand structure |

## ReAct Example: "Write a function that reverses a string"

```
[Reflect] User wants a string reversal function. I need to:
  - Write the function
  - Test it with examples
  - Show it works

[Plan] I will use run_python() to:
  1. Define the function
  2. Call it with test inputs
  3. Verify the output

[Tool Call] run_python("
def reverse(s):
    return s[::-1]

# Test it
print(reverse('hello'))      # 'olleh'
print(reverse('Python'))     # 'nohtyP'
")

[Observe] Output: 
  olleh
  nohtyP

[Respond] Here's your string reversal function:
  def reverse(s):
      return s[::-1]
  
  It works by using Python's slice notation s[::-1] 
  which means "from end to start, step -1".
```

## Best Practices
- Always test code with `run_python()` before returning it
- Use clear variable names and comments
- Show 2-3 examples of usage
- Explain the key algorithm or concept
- Keep code minimal, readable, and focused on learning
