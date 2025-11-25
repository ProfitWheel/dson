import sys
import os
from typing import List, Type

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from langchain_core.output_parsers import BaseOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel as LCBaseModel # LangChain often uses v1
    from pydantic import BaseModel
except ImportError:
    print("LangChain not installed. Please install with: pip install langchain langchain-core")
    sys.exit(0)

import src.dson as dson

# Define model (standard Pydantic)
class Product(BaseModel):
    id: int
    name: str
    tags: List[str]
    price: float

# Custom DSON Parser for LangChain
class DSONOutputParser(BaseOutputParser):
    pydantic_object: Type[BaseModel]

    def parse(self, text: str) -> BaseModel:
        # Clean up potential markdown code blocks which LLMs might add
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            if text.startswith("dson"):
                text = text[4:].strip()
        
        return dson.parse(text, self.pydantic_object)

    def get_format_instructions(self) -> str:
        return dson.format_instructions(self.pydantic_object)

def main():
    print("--- LangChain Integration Example ---")
    
    # 1. Setup Parser
    parser = DSONOutputParser(pydantic_object=Product)
    
    # 2. Create Prompt
    prompt = PromptTemplate(
        template="Extract product info.\n{format_instructions}\n\nInput: {query}\nOutput:",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    # 3. Simulate Chain Execution
    query = "I need a Super Widget for $99.99 tagged with 'pro' and 'new'"
    
    # This is what the prompt would look like sent to LLM
    final_prompt = prompt.format(query=query)
    print("\n[Prompt sent to LLM]:")
    print(final_prompt)
    
    # 4. Simulate LLM Response
    llm_response = "%PRO|101|Super Widget|pro|new||99.99"
    print(f"\n[LLM Response]: {llm_response}")
    
    # 5. Parse Output
    result = parser.parse(llm_response)
    print("\n[Parsed Result]:")
    print(result)
    print(f"Type: {type(result)}")

if __name__ == "__main__":
    main()
