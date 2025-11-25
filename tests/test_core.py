import pytest
from typing import List, Optional
from pydantic import BaseModel
import dson

class User(BaseModel):
    id: int
    name: str
    roles: List[str]

class Product(BaseModel):
    sku: str
    price: float
    tags: Optional[List[str]] = None

def test_schema_generation():
    instr = dson.format_instructions(User)
    print(instr)
    assert "%USE:id|name|roles[]" in instr
    assert "Sep='|'" in instr

def test_parsing_simple():
    text = "%USE|1|Alice|admin|editor||"
    user = dson.parse(text, User)
    assert user.id == 1
    assert user.name == "Alice"
    assert user.roles == ["admin", "editor"]

def test_parsing_dirty():
    text = "Here is the data:\n```\n%USE|1|Alice|admin|editor||\n```"
    user = dson.parse(text, User)
    assert user.id == 1
    assert user.name == "Alice"

def test_parsing_types():
    text = "%PRO|ABC|10.5|tag1||"
    p = dson.parse(text, Product)
    assert p.sku == "ABC"
    assert p.price == 10.5
    assert p.tags == ["tag1"]

def test_parsing_nulls():
    # Optional list
    text = "%PRO|ABC|10.5|~"
    p = dson.parse(text, Product)
    assert p.tags is None

def test_parsing_empty_list():
    text = "%USE|1|Alice||" # Empty list?
    # |Alice| -> next is roles.
    # If roles is empty, we expect || immediately?
    # If split by |, we get "Alice", "", "".
    # Parser logic: if list, consume until ||.
    # If next token is "", it breaks.
    user = dson.parse(text, User)
    assert user.roles == []
