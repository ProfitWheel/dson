# DSON Usage Guide

## Installation
```bash
pip install dson[all]
```

## 1. Define Your Model
Use standard Pydantic models - **no changes needed!**

```python
from typing import List
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    roles: List[str]
```

## 2. Generate Instructions
Inject the DSON schema into your system prompt.

```python
import dson

system_prompt = f"""
You are a data extractor.
{dson.format_instructions(User)}
"""
```

## 3. Parse Output
Parse the LLM response string directly into your object.

```python
llm_output = "%D|1|Alice|admin|editor||"
user = dson.parse(llm_output, User)
print(user.name) # Alice
```

### Advanced Parsing
DSON supports two tolerance modes:

- **`"boost"` (Default)**: Fault-tolerant. Strips preambles ("Here is..."), markdown blocks, and extra whitespace.
- **`"strict"`**: Requires exact DSON format starting with `%TAG`.

```python
# Boost mode (default) - handles dirty output
user = dson.parse(dirty_output, User, tolerance_mode="boost")

# Strict mode - enforces exact format
user = dson.parse(clean_output, User, tolerance_mode="strict")
```

## 4. Framework Integration

### LangChain
```python
from dson.adapters.langchain import DSONParser

parser = DSONParser(pydantic_object=User)
chain = prompt | model | parser
```

### LlamaIndex
```python
from dson.adapters.llamaindex import DSONProgram

program = DSONProgram.from_defaults(
    output_cls=User,
    llm=llm,
    prompt_template_str="Extract user data from: {text}"
)
user = program(text="Alice is an admin.")
```

## 5. Migration Tool
Scan your codebase for legacy JSON prompts and rewrite them.

```bash
# After pip install dson[all], use:
dson-migrate --folder ./src

# Auto-rewrite (requires OPENROUTER_API_KEY)
dson-migrate --folder ./src --auto
```

## 6. Performance Note: Cost Arbitrage

DSON is designed to optimize for **output tokens**, which are significantly more expensive (3-10x) than input tokens.

- **Input Overhead**: You will see a larger prompt due to the schema definition.
- **Output Savings**: You will see ~30-50% smaller responses.

For high-volume data extraction, this trade-off results in **lower overall costs** and **faster generation speeds**.
