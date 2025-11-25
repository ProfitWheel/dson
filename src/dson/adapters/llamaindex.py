from typing import Type, TypeVar, Optional, Any
try:
    from llama_index.core.types import BasePydanticProgram
    from llama_index.core import PromptTemplate
    from llama_index.core.llms import LLM
except ImportError:
    class BasePydanticProgram: pass
    class PromptTemplate: pass
    class LLM: pass

from pydantic import BaseModel
import dson

T = TypeVar("T", bound=BaseModel)

class DSONProgram(BasePydanticProgram):
    """A DSON-based Pydantic program for LlamaIndex."""
    
    def __init__(
        self,
        output_cls: Type[T],
        llm: LLM,
        prompt: PromptTemplate,
        tag: str = None,
        tolerance_mode: str = "boost",
        verbose: bool = False,
    ):
        self._output_cls = output_cls
        self._llm = llm
        self._prompt = prompt
        self._tag = tag
        self._tolerance_mode = tolerance_mode
        self._verbose = verbose

    @property
    def output_cls(self) -> Type[T]:
        return self._output_cls

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[T],
        llm: LLM,
        prompt_template_str: Optional[str] = None,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> "DSONProgram":
        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is None:
            prompt = PromptTemplate(prompt_template_str)
        
        return cls(output_cls=output_cls, llm=llm, prompt=prompt, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        fmt_instr = dson.format_instructions(self._output_cls, self._tag)
        formatted_prompt = self._prompt.format(**kwargs)
        full_prompt = f"{fmt_instr}\n\n{formatted_prompt}"
        
        response = self._llm.complete(full_prompt)
        return dson.parse(response.text, self._output_cls, self._tag, self._tolerance_mode)
        
    async def acall(self, *args: Any, **kwargs: Any) -> T:
        fmt_instr = dson.format_instructions(self._output_cls, self._tag)
        formatted_prompt = self._prompt.format(**kwargs)
        full_prompt = f"{fmt_instr}\n\n{formatted_prompt}"
        
        response = await self._llm.acomplete(full_prompt)
        return dson.parse(response.text, self._output_cls, self._tag, self._tolerance_mode)
