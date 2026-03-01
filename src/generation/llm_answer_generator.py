"""LLM-based answer generator implementation"""
from typing import List, Iterator
from src.models import RetrievedDocument
from src.config import get_config
from src.utils import get_logger
from src.prompts.generation import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_STREAMING_SYSTEM_PROMPT,
    answer_user_prompt,
    answer_streaming_user_prompt,
)
from openai import OpenAI
from .answer_generator import AnswerGenerator

logger = get_logger()


class LLMAnswerGenerator(AnswerGenerator):
    """Generate answers using LLM"""

    def __init__(self):
        config = get_config()
        self.client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )
        self.model = config.llm.model_name
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens

    def generate(self, query: str, context: List[RetrievedDocument]) -> str:
        """Generate answer from context (buffered)"""
        context_text = self._build_context(context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": answer_user_prompt(context_text, query)
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Unable to generate answer."

    def generate_streaming(self, query: str, context: List[RetrievedDocument]) -> Iterator[str]:
        """Stream answer token-by-token"""
        context_text = self._build_context(context)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANSWER_STREAMING_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": answer_streaming_user_prompt(context_text, query)
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"Error: {str(e)}"

    @staticmethod
    def _build_context(docs: List[RetrievedDocument]) -> str:
        """Build context string from retrieved documents, preferring parent chunks when available"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Use parent content if available (for parent-child retrieval), otherwise use regular content
            content = doc.parent_content if doc.parent_content else doc.content
            context_parts.append(f"Source {i} ({doc.source}):\n{content}\n")
        return "\n".join(context_parts)
