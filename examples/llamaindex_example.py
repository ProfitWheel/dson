import sys
import os
from typing import Type, Any
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Try newer import structure (v0.10+)
    from llama_index.core.output_parsers.base import BaseOutputParser
except ImportError:
    try:
        # Try older import structure
        from llama_index.output_parsers.base import BaseOutputParser
    except ImportError:
        print("LlamaIndex not installed. Please install with: pip install llama-index-core")
        sys.exit(0)

import src.dson as dson

class Book(BaseModel):
    title: str
    author: str
    year: int

class DSONParser(BaseOutputParser):
    def __init__(self, pydantic_object: Type[BaseModel]):
        self._pydantic_object = pydantic_object
        
    def parse(self, output: str) -> Any:
        # Clean up potential markdown
        output = output.strip()
        if output.startswith("```") and output.endswith("```"):
            output = output[3:-3].strip()
            if output.startswith("dson"):
                output = output[4:].strip()
                
        return dson.parse(output, self._pydantic_object)

    def format(self, query: str) -> str:
        # This method is used to inject instructions into the prompt
        return query + "\n\n" + dson.format_instructions(self._pydantic_object)

def main():
    print("--- LlamaIndex Integration Example ---")
    
    parser = DSONParser(Book)
    
    # Simulate LLM response
    llm_output = "%BOO|The Great Gatsby|F. Scott Fitzgerald|1925"
    print(f"LLM Output: {llm_output}")
    
    result = parser.parse(llm_output)
    print(f"Parsed: {result}")
    print(f"Title: {result.title}")

if __name__ == "__main__":
    main()
