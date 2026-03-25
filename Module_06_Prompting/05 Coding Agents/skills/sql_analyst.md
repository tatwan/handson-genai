# Skill: SQL Analyst

## Purpose
Write and run SQL queries against a local SQLite database using Python. Ideal for students learning databases or wanting to move beyond CSV files.

## When to Use This Skill
- "Query this database and find the top 5 customers by revenue"
- "How many orders were placed in January?"
- "Join these two tables and show the result"
- "Create a summary table from this dataset"

## Available Tools
| Tool | Purpose |
|------|---------|
| `run_python(code)` | Execute SQL via Python's `sqlite3` module |
| `read_file(filepath)` | Read a `.sql` schema or data file |

## Setup Pattern (Always Start Here)

Every SQL task needs to connect to a database. Use this pattern at the start:

```python
import sqlite3
conn = sqlite3.connect("my_database.db")   # creates file if it doesn't exist
cursor = conn.cursor()
```

For quick demos, create an in-memory database (nothing written to disk):
```python
conn = sqlite3.connect(":memory:")         # in-memory, reset every run
```

## ReAct Example: "I have an orders table. Find the top 3 customers by total spend."

```
[Reflect] User wants to aggregate an orders table by customer.
  - Need to: GROUP BY customer, SUM the amounts, ORDER BY total DESC, LIMIT 3
  - I don't know the table schema yet — must inspect first

[Plan]
  Step 1: Create a sample orders table (or inspect existing schema)
  Step 2: Run the aggregation query
  Step 3: Format and explain the result

[Tool Call #1] run_python("
import sqlite3

# Create in-memory database with sample data
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create and populate orders table
cursor.executescript('''
  CREATE TABLE orders (
    order_id   INTEGER PRIMARY KEY,
    customer   TEXT,
    product    TEXT,
    amount     REAL,
    order_date TEXT
  );

  INSERT INTO orders VALUES
    (1, 'Alice',   'Laptop',  1200, '2024-01-05'),
    (2, 'Bob',     'Mouse',     25, '2024-01-06'),
    (3, 'Alice',   'Monitor',  350, '2024-01-08'),
    (4, 'Charlie', 'Keyboard',  80, '2024-01-09'),
    (5, 'Bob',     'Laptop',  1200, '2024-01-10'),
    (6, 'Alice',   'Webcam',   120, '2024-01-12');
''')

# Inspect the schema first
cursor.execute(\"PRAGMA table_info(orders)\")
print('Schema:', [row[1] for row in cursor.fetchall()])
conn.commit()
print('Sample rows:')
for row in cursor.execute('SELECT * FROM orders LIMIT 3'):
    print(' ', row)
")

[Observe] Output:
  Schema: ['order_id', 'customer', 'product', 'amount', 'order_date']
  Sample rows:
    (1, 'Alice', 'Laptop', 1200, '2024-01-05')
    (2, 'Bob', 'Mouse', 25, '2024-01-06')
    (3, 'Alice', 'Monitor', 350, '2024-01-08')

[Tool Call #2] run_python("
import sqlite3
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
cursor.executescript('''
  CREATE TABLE orders (order_id INTEGER, customer TEXT, product TEXT, amount REAL, order_date TEXT);
  INSERT INTO orders VALUES (1,'Alice','Laptop',1200,'2024-01-05'),
    (2,'Bob','Mouse',25,'2024-01-06'),(3,'Alice','Monitor',350,'2024-01-08'),
    (4,'Charlie','Keyboard',80,'2024-01-09'),(5,'Bob','Laptop',1200,'2024-01-10'),
    (6,'Alice','Webcam',120,'2024-01-12');
''')

# The real query: TOP 3 customers by total spend
query = '''
  SELECT
    customer,
    COUNT(*)        AS orders_placed,
    SUM(amount)     AS total_spend
  FROM orders
  GROUP BY customer
  ORDER BY total_spend DESC
  LIMIT 3
'''
print(f'{'Customer':<12} {'Orders':>8} {'Total Spend':>12}')
print('-' * 34)
for row in cursor.execute(query):
    print(f'{row[0]:<12} {row[1]:>8} \${row[2]:>11,.2f}')
")

[Observe] Output:
  Customer      Orders  Total Spend
  ----------------------------------
  Alice              3     $1,670.00
  Bob                2     $1,225.00
  Charlie            1        $80.00

[Respond] Top 3 customers by total spend:
  1. Alice — $1,670 across 3 orders
  2. Bob — $1,225 across 2 orders
  3. Charlie — $80 (1 order)
```

## Best Practices
- **Inspect schema before querying** — use `PRAGMA table_info(table_name)`
- Use `sqlite3` (built into Python, no install needed)
- Always `conn.commit()` after INSERT/UPDATE/DELETE
- Use `:memory:` for demos; use a `.db` filename to persist data
- Format output with aligned columns for readability
- Show the SQL query alongside results — the query IS the explanation

## Key SQL Patterns

```sql
-- Aggregation
SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer;

-- Filtering
SELECT * FROM orders WHERE amount > 500;

-- Joining two tables
SELECT o.customer, p.category
FROM orders o
JOIN products p ON o.product = p.name;

-- Date filtering
SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31';
```
