"""
Generic LLM adapter for evaluation harness.

Any metric needing an LLM to return validated, schema-shaped output
depend on this Protocol. 
Never a concrete LLM client -- mirrors ragregress.rag.interface.RAGSystem. 

Prompts/schemas/methodology stay owned by the metric; StructuredLLM only executes.


INTERNAL SEAM, not a supported extension point (D13 J6): the lab builds its own judge from
`judge_model` in the run config (see ragregress.evaluation.judge). The Protocol exists so metrics
and tests can swap in a fake judge, not so users can bring their own.
"""

from typing import Optional, Protocol, Type, TypeVar

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
                # print(f"[DEBUG] schema={schema.__name__} result={result!r}")
                return result
                
            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"LangChainStructuredLLM: failed to get valid {schema.__name__} "
            f"after {self.max_retries + 1} attempt(s)"
        ) from last_error


# Built in example with Instructor:

class InstructorStructuredLLM:
    """
        An additional StructuredLLM implementation using Instructor over an OpenAI-compatible endpoint.
        It supports reask behavior; LangChainStructuredLLM remains the default elsewhere. 
        Switching only changes the llm passed to metrics, not Metric, MetricResult, or evaluate().

        provider is "ollama" or "openai". base_url and api_key come from resolve_judge() (environment only, never run config)
        and are passed to Instructor only when set. 
        
        Without them, InstructorStructuredLLM("model") behaves as before. 
        Ollama ignores api_key and always sends "ollama"."""

    def __init__(self, model_name: str, provider: str = "ollama", max_retries: int = 2, seed: Optional[int] = None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None):
        import instructor

        self.model_name = model_name
        self.provider = provider
        self.max_retries = max_retries
        self.seed = seed
 
        client_kwargs = {}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        self.client = instructor.from_provider(f"{provider}/{model_name}", mode=instructor.Mode.JSON, **client_kwargs)

    def generate_structured(self, prompt: str, schema: Type[SchemaT]) -> SchemaT:
        extra = {} if self.seed is None else {"seed": self.seed}
        result = self.client.chat.completions.create(
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_retries=self.max_retries,
            **extra,
        )
 
        return result
        