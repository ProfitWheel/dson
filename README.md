# DSON (Dense Sequence Object Notation)

**DSON** is a schema-first, delimiter-separated protocol designed to replace JSON in LLM workflows, reducing token usage by 30-50%.

## Key Features
- **Schema-First**: Schema is sent once in system prompt, not repeated in every object.
- **Fault-Tolerant**: "Boost" mode handles dirty LLM output (markdown, preambles) gracefully.
- **Type-Safe**: Validates output against your Pydantic models.
- **Framework Ready**: Drop-in adapters for LangChain and LlamaIndex.

## Zero-Friction Installation

```bash
pip install dson
# Extras
pip install dson[langchain]
pip install dson[llamaindex]
pip install dson[benchmark]
pip install dson[all]
```

## Quick Start (No Code Changes Required!)

```python
from pydantic import BaseModel  # Use your existing models!
from typing import List
import dson

# Your existing Pydantic model - NO CHANGES NEEDED
class User(BaseModel):
    id: int
    name: str
    roles: List[str]

# Generate DSON instructions for your prompt
system_prompt = f"""
Extract user data.
{dson.format_instructions(User)}
"""

# Parse LLM response
llm_output = "%D|1|Alice|admin|editor||"
user = dson.parse(llm_output, User)
print(user.name)  # Alice
```

## Why DSON?

**30-50% Output Token Savings** - Optimized for LLM data extraction tasks

The benchmark measures DSON's performance in **extraction tasks** where LLMs convert raw data (CSV, text, tables) into structured formats.

**Key Metrics:**
- **Parsable (Exact Match)**: Output parsed correctly AND matches ground truth 100%
- **Accuracy (Fuzzy Match)**: Score 0.0-1.0 showing how close output is to expected data
- **Input Overhead**: Extra tokens used in prompt (cheaper tokens)
- **Output Savings**: Reduction in generated tokens (expensive tokens)

**Running Benchmarks:**
```bash
# Run full benchmark (5 models × 3 data types × 10 examples)
python benchmarks/run_bench.py

# Quick sanity check
python benchmarks/run_bench.py --sanity

# Custom configuration
python benchmarks/run_bench.py --models gpt-4o --dtypes Tabular --examples 5
```

**Output Files:**
- `results_*.csv` - Detailed per-run data with 3 rows per test (JSON, DSON-boost, DSON-strict)
- `summary_*.txt` - Aggregate metrics and per-model breakdown
- `benchmarks.db` - SQLite database for analysis

**Benchmark compares:**
1. **JSON** - Standard JSON format (baseline)
2. **DSON (boost mode)** - Fault-tolerant parsing
3. **DSON (strict mode)** - Exact syntax enforcement

See [benchmarks/results/README.md](benchmarks/results/README.md) for latest results.

### Cost Arbitrage

DSON is designed to **trade cheap input tokens for expensive output tokens**.

| Metric | DSON | JSON | Impact |
|--------|------|------|--------|
| **Input Overhead** | **High** | Low | You pay slightly more for prompt (schema definition) |
| **Output Savings** | **~30-50%** | 0% | You save significantly on generation (expensive) |
| **Net Cost** | **Lower** | Higher | **Overall cost reduction for high-volume tasks** |

**Tag Optimization**: Benchmarks use `%D` (2 chars) instead of model-based tags to maximize token savings. In normal usage, tags are auto-generated from your model name (e.g., `User` → `%USE`).

## Documentation

- [USAGE.md](USAGE.md) - Detailed usage guide
- [benchmarks/results/README.md](benchmarks/results/README.md) - Latest benchmark results
- [whitepaper.md](whitepaper.md) - Technical deep dive

## License

MIT
