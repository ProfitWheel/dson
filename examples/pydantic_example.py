from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add project root to path to import dson
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.dson as dson

# 1. Define your Pydantic model
class User(BaseModel):
    id: int
    name: str
    email: str
    roles: List[str]
    is_active: bool = True

# 2. Generate format instructions for your prompt
instructions = dson.format_instructions(User)
print("--- System Prompt Instructions ---")
print(instructions)
print("----------------------------------")

# 3. Simulate LLM Output (DSON format)
# In a real app, this would come from the LLM
llm_output = "%USE|1|Alice Smith|alice@example.com|admin|editor||True"

print(f"\n--- Raw LLM Output ---\n{llm_output}\n----------------------")

# 4. Parse the output
try:
    user = dson.parse(llm_output, User)
    print("\n--- Parsed Pydantic Object ---")
    print(f"Name: {user.name}")
    print(f"Email: {user.email}")
    print(f"Roles: {user.roles}")
    print(f"Active: {user.is_active}")
    print(f"Full Dump: {user.model_dump()}")
    print("------------------------------")
except Exception as e:
    print(f"Parsing failed: {e}")
