import typing
from typing import Type, Any, List, get_origin, get_args
from pydantic import BaseModel

def _unwrap_optional(t: Any) -> Any:
    origin = get_origin(t)
    if origin is typing.Union:
        args = get_args(t)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return t

def _get_type_name(t: Any) -> str:
    t = _unwrap_optional(t)
    origin = get_origin(t)
    if origin is list or origin is List:
        return f"{_get_type_name(get_args(t)[0])}[]"
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t)

def _generate_schema_fields(model: Type[BaseModel], prefix: str = "") -> List[str]:
    fields = []
    for name, field in model.model_fields.items():
        # Check if nested Pydantic model
        annotation = _unwrap_optional(field.annotation)
        origin = get_origin(annotation)
        
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            # Nested model - flatten
            fields.extend(_generate_schema_fields(annotation, prefix=f"{prefix}{name}."))
        else:
            # Simple field
            fields.append(f"{prefix}{name}")
            if origin is list or origin is List:
                fields[-1] += "[]"
    return fields

def generate_schema_string(model: Type[BaseModel], tag: str) -> str:
    fields = _generate_schema_fields(model)
    return f"{tag}:{'|'.join(fields)}"

def get_format_instructions(model: Type[BaseModel], tag: str) -> str:
    schema = generate_schema_string(model, tag)
    
    # Better example that shows null and works for all types
    example_str = f"{tag}|1|Alice|Hardware|99.5|~"
    

    # STRENGTHENED PROMPT - eliminates explanatory text
    strict_rules = (
        f"FORMAT: CUSTOM\n"
        f"SCHEMA: {schema}\n"
        f"RULES: Sep='|' ArrEnd='||' Null='~'\n"
        f"EX: {example_str}\n\n"
        f"CONSTRAINTS:\n"
        f"1 OUTPUT MUST START WITH {tag} - NO OTHER TEXT BEFORE IT\n"
        f"2 ZERO explanations. ZERO markdown blocks. ZERO preamble.\n"
        f"3 Start immediately: {tag}|value1|value2|...\n\n"
        f"CORRECT: {tag}|value1|value2|..."
    )
    return strict_rules.strip()
