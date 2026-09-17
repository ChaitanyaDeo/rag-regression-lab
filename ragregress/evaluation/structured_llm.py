"""
Generic LLM adapter for evaluation harness.

Any metric needing an LLM to return validated, schema-shaped output
depend on this Protocol. 
Never a concrete LLM client -- mirrors ragregress.rag.interface.RAGSystem. 

Prompts/schemas/methodology stay owned by the metric; StructuredLLM only executes.
"""

from typing import Protocol, Type, TypeVar

from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredLLM(Protocol):
    def generate_structured(self, prompt: str, schema: Type[SchemaT])->SchemaT:
        """Given a prompt and a Pydantic schema, return a validated instance of it."""
        ...


# A builtin example

class LangChainStructuredLLM:
    """Built-in default StructuredLLM, wrapping any LangChain chat model
        via with_structured_output(). 
        Retries on failure (with_structured_output()
        has no retry of its own, unlike RAGAS's instructor-based approach --
        not adopted, to avoid a second LLM-orchestration stack).
    """

    def __init__(self, chat_model: BaseChatModel, max_retries: int = 1):
        self.chat_model = chat_model
        self.max_retries = max_retries

    def generate_structured(self, prompt:str, schema: Type[SchemaT])->SchemaT:
        structured_model = self.chat_model.with_structured_output(schema)


        last_error = None
        for _ in range(self.max_retries+1):
            try:
                result = structured_model.invoke(prompt)
                # raw_ans = self.chat_model.invoke(prompt, )
                print(f"[DEBUG] schema={schema.__name__} result={result!r}")
                return result
                
            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"LangChainStructuredLLM: failed to get valid {schema.__name__} "
            f"after {self.max_retries + 1} attempt(s)"
        ) from last_error


# Built in example with Instructor:

class InstructorStructuredLLM:
    """Built-in default StructuredLLM, wrapping any Instructor client chat model
    This is an ADDITIONAL implementation of the StructuredLLM protocol,
    not a replacement -- LangChainStructuredLLM is still used wherever
    this reask behavior isn't needed. Swapping between them only changes
    what's passed to e.g. Faithfulness(llm=...); nothing about Metric,
    MetricResult, or evaluate() changes.
    """

    def __init__(self, model_name: str, provider: str = "ollama", max_retries: int = 2):
        import instructor

        self.model_name = model_name
        self.max_retries = max_retries
        self.client = instructor.from_provider(f"{provider}/{model_name}", mode=instructor.Mode.JSON)

    def generate_structured(self, prompt: str, schema: Type[SchemaT]) -> SchemaT:
        result = self.client.chat.completions.create(
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_retries=self.max_retries,
        )

        return result
        