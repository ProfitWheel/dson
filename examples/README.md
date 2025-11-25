# DSON Integration Examples

This directory contains examples of how to use DSON with popular frameworks.

## 1. Pydantic Core (`pydantic_example.py`)
Shows the most basic usage:
- Defining a Pydantic model
- Generating DSON instructions
- Parsing DSON output back into the model

**Run:**
```bash
python examples/pydantic_example.py
```

## 2. LangChain (`langchain_example.py`)
Shows how to create a custom `BaseOutputParser` for LangChain.
- Integrates with LangChain's `PromptTemplate`
- Parses output into Pydantic objects

**Run:**
```bash
pip install langchain langchain-core
python examples/langchain_example.py
```

## 3. LlamaIndex (`llamaindex_example.py`)
Shows how to create a custom `BaseOutputParser` for LlamaIndex.
- Compatible with LlamaIndex query engines
- Handles DSON parsing logic

**Run:**
```bash
pip install llama-index-core
python examples/llamaindex_example.py
```
