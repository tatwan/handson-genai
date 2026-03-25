# Skill: Math Solver

## Purpose
Solve mathematical problems step-by-step, verify computations with code, and explain the reasoning clearly. Makes the agent accessible to students without a coding background.

## When to Use This Skill
- "Solve this equation: 2x + 5 = 13"
- "What is the compound interest on $1,000 at 5% over 10 years?"
- "Is 1024 a power of 2?"
- "Calculate the standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]"

## Available Tools
| Tool | Purpose |
|------|---------|
| `run_python(code)` | Compute and verify numeric results |

## ReAct Example: "What is the compound interest on $1,000 at 5% annual rate for 3 years?"

```
[Reflect] This is a compound interest problem.
  - Formula: A = P(1 + r/n)^(nt)
    where P=principal, r=rate, n=compounds/year, t=years
  - Assuming annual compounding (n=1)

[Plan]
  1. Write out the formula with the given values
  2. Compute with run_python() to get an exact answer
  3. Show both the formula step and the result clearly

[Tool Call] run_python("
P = 1000     # principal ($)
r = 0.05     # annual rate (5%)
n = 1        # compounded once per year
t = 3        # years

# Compound interest formula
A = P * (1 + r/n) ** (n * t)
interest = A - P

print(f'Principal:         ${P:,.2f}')
print(f'Rate:              {r*100:.1f}% per year')
print(f'Time:              {t} years')
print(f'Final amount:      ${A:,.2f}')
print(f'Interest earned:   ${interest:,.2f}')
")

[Observe] Output:
  Principal:         $1,000.00
  Rate:              5.0% per year
  Time:              3 years
  Final amount:      $1,157.63
  Interest earned:   $157.63

[Respond] Using compound interest formula A = P(1 + r)^t:
  A = 1000 × (1.05)³ = $1,157.63
  Interest earned: $157.63 over 3 years.
```

## ReAct Example 2: "Solve 2x + 5 = 13"

```
[Reflect] Linear equation in one variable.
  - Isolate x: subtract 5 from both sides → 2x = 8
  - Divide by 2 → x = 4
  - Verify by substituting back

[Tool Call] run_python("
# Show step-by-step solution
print('Equation: 2x + 5 = 13')
print('Step 1:   2x = 13 - 5 = 8')
print('Step 2:   x = 8 / 2 = 4')
print()

# Verify
x = 4
result = 2*x + 5
print(f'Verification: 2({x}) + 5 = {result}  ✓' if result == 13 else 'Verification FAILED')
")

[Observe] Output:
  Equation: 2x + 5 = 13
  Step 1:   2x = 13 - 5 = 8
  Step 2:   x = 8 / 2 = 4
  Verification: 2(4) + 5 = 13  ✓

[Respond] x = 4. Verified: 2(4) + 5 = 13 ✓
```

## Best Practices
- Always **show the formula before plugging in numbers** — the thinking matters as much as the answer
- Use `run_python()` to compute, even for "simple" arithmetic — avoids hallucination errors
- Print clear labels (`Principal`, `Rate`, etc.) not just raw numbers
- Round to 2 decimal places for financial results; show full precision for math
- If the problem has multiple interpretations (e.g., simple vs. compound interest), state your assumption
