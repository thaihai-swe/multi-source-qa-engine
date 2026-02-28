"""Query expansion module for generating alternative query formulations"""
from typing import List
from src.config import get_config
from src.utils import get_logger
from src.prompts.reasoning import (
    QUERY_EXPANSION_SYSTEM_PROMPT,
    query_expansion_user_prompt,
)
from openai import OpenAI

logger = get_logger()


class QueryExpander:
    """Expands queries into variations for improved retrieval coverage"""

    @staticmethod
    def expand(query: str, num_variations: int = 4) -> List[str]:
        """Generate query variations using LLM"""
        logger.info(f"🔄 Generating {num_variations} query variations...")

        config = get_config()
        client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key
        )

        try:
            prompt = query_expansion_user_prompt(query, num_variations)

            response = client.chat.completions.create(
                model=config.llm.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": QUERY_EXPANSION_SYSTEM_PROMPT
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            variations_text = response.choices[0].message.content.strip()
            variations = [v.strip() for v in variations_text.split('\n') if v.strip()]

            # Always include original query
            variations = [query] + variations[:num_variations-1]
            logger.info(f"✅ Generated {len(variations)} variations")
            return variations

        except Exception as e:
            logger.warning(f"⚠️ Query expansion failed: {str(e)}")
            return [query]  # Fallback to original
