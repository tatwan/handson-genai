# Skill: Data Analyst

## Purpose
Analyze, summarize, and answer questions about structured data (CSV files, tables, datasets).

## When to Use This Skill
- "Which product has the highest revenue?"
- "Summarize the data in sales.csv"
- "What's the average price?"
- "Analyze this dataset"

## Available Tools
| Tool | Purpose |
|------|---------|
| `read_file(filepath)` | Read CSV, TXT, or data files |
| `run_python(code)` | Run pandas or analysis code |

## ReAct Example: "I have sales.csv. Which product has the highest revenue?"

```
[Reflect] User wants to know the top product by revenue.
  - I need to read sales.csv first
  - Inspect the columns and data
  - Find the product with max revenue

[Plan] Two-step approach:
  1. Use read_file() to inspect the data
  2. Use run_python() with pandas to find the answer

[Tool Call #1] read_file("sales.csv")

[Observe] 
  product,revenue,units_sold
  Widget A,4200,210
  Widget B,1800,90
  Gadget X,6500,130
  Gadget Y,2100,70

[Tool Call #2] run_python("
import pandas as pd
df = pd.read_csv('sales.csv')
top_product = df.loc[df['revenue'].idxmax()]
print(f\"Top product: {top_product['product']}\")
print(f\"Revenue: ${top_product['revenue']}\")
")

[Observe] Output:
  Top product: Gadget X
  Revenue: $6500

[Respond] **Gadget X** has the highest revenue at **$6,500**.
```

## Best Practices
- Always inspect the file structure first (columns, data types)
- Show column names and the first few rows before analysis
- Use pandas for structured data (CSV, tabular)
- Round numeric results to 2 decimal places
- Explain what the numbers mean (context matters!)
- Support follow-up questions about the same dataset
