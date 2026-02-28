"""Centralized prompt management for the RAG system.

All LLM prompts are defined here so they can be maintained, versioned,
and reused from a single location.

Modules
-------
- generation   : Answer-generation prompts (system + user).
- reasoning    : Query expansion, multi-hop, self-query decomposition.
- evaluation   : RAGAS metrics, fact-checking, hallucination detection.
- retrieval    : Domain-guard topic extraction and relevance scoring.
"""

from src.prompts.generation import *   # noqa: F401,F403
from src.prompts.reasoning import *    # noqa: F401,F403
from src.prompts.evaluation import *   # noqa: F401,F403
from src.prompts.retrieval import *    # noqa: F401,F403
