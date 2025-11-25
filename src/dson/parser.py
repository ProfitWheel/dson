import re
import typing
from typing import Type, TypeVar, Any, Dict
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class DSONParsingError(Exception):
    pass

def _clean_text(text: str, tolerance_mode: str = "boost") -> str:
    text = re.sub(r'```(?:dson)?\n?', '', text)
    text = re.sub(r'```', '', text)
    if tolerance_mode == "boost":
        text = re.sub(r'^(?:Here (?:is|are) the (?:data|output|result)s?:?\s*)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(?:Output:?\s*)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\|\s*', '|', text)
    return text.strip()

def _tokenize(text: str, tolerance_mode: str = "boost") -> list:
    tokens = text.split('|')
    if tolerance_mode == "boost":
        cleaned = []
        for token in tokens:
            cleaned.append(token.strip() if token else token)
        return cleaned
    return tokens

def _unwrap_optional(t: Any) -> Any:
    origin = typing.get_origin(t)
    if origin is typing.Union:
        args = typing.get_args(t)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return t

def _coerce_type(value: str, target_type: Any, tolerance_mode: str = "boost") -> Any:
    if tolerance_mode != "boost":
        return value
    if value is None:
        return None
    try:
        if target_type == int:
            return int(float(value))
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return value.lower() in ('true', 't', '1', 'yes')
        elif target_type == str:
            return value
    except (ValueError, AttributeError):
        pass
    return value

def parse_dson(text: str, model: Type[T], tolerance_mode: str = "boost") -> T:
    clean_text = _clean_text(text, tolerance_mode)
    if not clean_text.startswith("%"):
        if tolerance_mode == "boost":
            match = re.search(r'%\w+', clean_text)
            if match:
                clean_text = clean_text[match.start():]
            else:
                raise DSONParsingError(f"Invalid DSON: Missing start tag. Got: {clean_text[:20]}")
        else:
            raise DSONParsingError(f"Invalid DSON: Missing start tag. Got: {clean_text[:20]}")
    
    tokens = _tokenize(clean_text, tolerance_mode)
    tag = tokens.pop(0)
    
    def consume_model(m: Type[BaseModel], current_idx: int):
        local_data = {}
        c_idx = current_idx
        for name, field in m.model_fields.items():
            annotation = _unwrap_optional(field.annotation)
            origin = typing.get_origin(annotation)
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                nested_data, new_idx = consume_model(annotation, c_idx)
                local_data[name] = nested_data
                c_idx = new_idx
            elif origin is list:
                items = []
                item_type = typing.get_args(annotation)[0]
                is_item_optional = _unwrap_optional(item_type) != item_type
                while c_idx < len(tokens):
                    val = tokens[c_idx]
                    c_idx += 1
                    if val == "":
                        break
                    if val == "~":
                        if len(items) == 0 and not is_item_optional:
                            local_data[name] = None
                            items = None
                            break
                        items.append(None)
                    else:
                        items.append(val)
                if items is not None:
                    local_data[name] = items
            else:
                if c_idx >= len(tokens):
                    val = "~"
                else:
                    val = tokens[c_idx]
                    c_idx += 1
                if val == "~":
                    local_data[name] = None
                else:
                    coerced_val = _coerce_type(val, annotation, tolerance_mode)
                    local_data[name] = coerced_val
        return local_data, c_idx
    
    try:
        data_dict, _ = consume_model(model, 0)
        return model(**data_dict)
    except Exception as e:
        raise DSONParsingError(f"Failed to hydrate object: {e}")

def parse_dson_list(text: str, model: Type[T], tolerance_mode: str = "boost") -> list:
    clean_text = _clean_text(text, tolerance_mode)
    tokens = _tokenize(clean_text, tolerance_mode)
    results = []
    idx = 0
    
    def consume_model(m: Type[BaseModel], current_idx: int):
        local_data = {}
        c_idx = current_idx
        for name, field in m.model_fields.items():
            annotation = _unwrap_optional(field.annotation)
            origin = typing.get_origin(annotation)
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                nested_data, new_idx = consume_model(annotation, c_idx)
                local_data[name] = nested_data
                c_idx = new_idx
            elif origin is list:
                items = []
                item_type = typing.get_args(annotation)[0]
                is_item_optional = _unwrap_optional(item_type) != item_type
                while c_idx < len(tokens):
                    val = tokens[c_idx]
                    c_idx += 1
                    if val == "":
                        break
                    if val == "~":
                        if len(items) == 0 and not is_item_optional:
                            local_data[name] = None
                            items = None
                            break
                        items.append(None)
                    else:
                        items.append(val)
                if items is not None:
                    local_data[name] = items
            else:
                if c_idx >= len(tokens):
                    val = "~"
                else:
                    val = tokens[c_idx]
                    c_idx += 1
                if val == "~":
                    local_data[name] = None
                else:
                    coerced_val = _coerce_type(val, annotation, tolerance_mode)
                    local_data[name] = coerced_val
        return local_data, c_idx
    
    # Loop to consume multiple objects
    while idx < len(tokens):
        # Skip empty tokens from || separators
        while idx < len(tokens) and tokens[idx] == "":
            idx += 1
        
        if idx >= len(tokens):
            break
            
        current_token = tokens[idx]
        
        # If the token looks like a tag (starts with %), consume it
        if current_token.startswith("%"):
            idx += 1
        elif tolerance_mode == "boost":
            pass
        
        try:
            data_dict, new_idx = consume_model(model, idx)
            results.append(model(**data_dict))
            idx = new_idx
            
        except Exception as e:
            if tolerance_mode == "strict":
                raise e
            else:
                break
                
    return results
