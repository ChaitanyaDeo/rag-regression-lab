"""
No RAG-baseline: a RAGSystem implementation that generates answers from the
model's own parametric knowledge, with no retrieval and no context.

This is seperate class implementing RAGSystem interface direclty.

The end user does not need to include this. It is purely for testing 
labs harness capability.

Reuses GeneratorConfig from generator.py (same model_name/temperature
knobs) since the LLM-calling mechanics are shared even though the
pipelines aren't.

"""

from typing import List
 
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from ragregress.rag.generator import GeneratorConfig
from ragregress.rag.interface import Chunk, GenerationResult

NO_CONTEXT_PROMPT = PromptTemplate.from_template(
    """Answer the question directly and concisely, using your own knowledge.
 
Question: {question}
 
Answer:"""
)

def run_generation_no_context(question: str, config: GeneratorConfig)->str:
    """Build the no-context prompt, call the LLM, return its raw text output.

    Mirrors generator.py's run_generation(), minus the context block
    """

    llm = ChatOllama(model=config.model_name, temperature=config.temperature)
    chain = NO_CONTEXT_PROMPT | llm
 
    response = chain.invoke({"question": question})

    return response.content

class NoRAGBaseline():
    """
    A closed RAGsystem baseline: ONLY generation

    NO Retrival, NO Context, NO Citation
    """

    def __init__(self, generator_config: GeneratorConfig = None):
        self.generator_config = generator_config or GeneratorConfig()


    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        """Always return [] as this is not rag 
        Just generation
        """
        return []

    def generate(self, query: str, contexts: List[Chunk]) -> GenerationResult:
        raw_output = run_generation_no_context(query, self.generator_config)

        answer = " ".join(raw_output.split())
        return GenerationResult(raw_output=raw_output, answer=answer, citations=[])