from typing import Type, TypeVar
try:
    from langchain_core.output_parsers import BaseOutputParser
except ImportError:
    class BaseOutputParser:
        pass

from pydantic import BaseModel
import dson

T = TypeVar("T", bound=BaseModel)

class DSONParser(BaseOutputParser):
    """LangChain OutputParser that parses DSON-formatted LLM responses into Pydantic objects."""
    
    def __init__(self, pydantic_object: Type[T], tag: str = None, tolerance_mode: str = "boost"):
        self.pydantic_object = pydantic_object
        self.tag = tag
        self.tolerance_mode = tolerance_mode

    def get_format_instructions(self) -> str:
        return dson.format_instructions(self.pydantic_object, self.tag)

    def parse(self, text: str) -> T:
        return dson.parse(text, self.pydantic_object, self.tag, self.tolerance_mode)

    @property
    def _type(self) -> str:
        return "dson_parser"
