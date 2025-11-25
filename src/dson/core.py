from typing import Type, TypeVar
from pydantic import BaseModel
from .instructions import get_format_instructions
from .parser import parse_dson

T = TypeVar("T", bound=BaseModel)

def _generate_tag(model: Type[BaseModel]) -> str:
    """Return standard DSON tag."""
    return "%D"

def format_instructions(model: Type[BaseModel], tag: str = None) -> str:
    """Generate DSON format instructions for a Pydantic model."""
    if tag is None:
        tag = _generate_tag(model)
    return get_format_instructions(model, tag)

def parse(text: str, model: Type[T], tag: str = None, tolerance_mode: str = "boost") -> T:
    """Parse DSON response into a Pydantic model instance.
    
    Args:
        text: Raw DSON response from LLM
        model: Pydantic model class
        tag: Optional custom tag (auto-generated if not provided)
        tolerance_mode: 'strict' for exact parsing, 'boost' for fault tolerance
    """
    if tag is None:
        tag = _generate_tag(model)
    return parse_dson(text, model, tolerance_mode)

def parse_list(text: str, model: Type[T], tag: str = None, tolerance_mode: str = "boost") -> list:
    """Parse DSON response into a list of Pydantic model instances.
    
    Args:
        text: Raw DSON response from LLM
        model: Pydantic model class
        tag: Optional custom tag (auto-generated if not provided)
        tolerance_mode: 'strict' for exact parsing, 'boost' for fault tolerance
    """
    if tag is None:
        tag = _generate_tag(model)
    # Note: parse_dson_list handles the tag internally by looking for it in the token stream
    from .parser import parse_dson_list
    return parse_dson_list(text, model, tolerance_mode)
