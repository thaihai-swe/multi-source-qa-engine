"""Self-query decomposition for complex multi-faceted questions.

Unlike MultiHopReasoner (sequential dependency chains), SelfQueryDecomposer
detects when a query asks about multiple *independent* aspects and answers
each sub-query separately before synthesising a unified response.

Example:
  Complex: "What caused WWI and how did it affect economic policy AND what
            were the resulting territorial changes?"
  → 3 independent sub-queries answered in parallel, then merged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from openai import OpenAI

from src.config import get_config
from src.prompts.reasoning import (
    self_query_complexity_prompt,
    self_query_decompose_prompt,
    self_query_synthesize_prompt,
)
from src.utils import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SubQuery:
    """A single independent sub-query produced by decomposition."""
    query: str
    aspect: str
    answer: Optional[str] = None


@dataclass
class SelfQueryDecomposition:
    """Full decomposition result including optional synthesised answer."""
    original_query: str
    is_complex: bool
    sub_queries: List[SubQuery]
    synthesized_answer: Optional[str] = None
    decomposition_method: str = "self_query"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "is_complex": self.is_complex,
            "sub_queries": [
                {"query": sq.query, "aspect": sq.aspect, "answer": sq.answer}
                for sq in self.sub_queries
            ],
            "synthesized_answer": self.synthesized_answer,
            "decomposition_method": self.decomposition_method,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class SelfQueryDecomposer:
    """Detects and decomposes complex multi-aspect queries.

    Workflow:
      1. ``analyze_complexity`` — decide whether the query has ≥2 independent
         aspects worth separating.
      2. ``decompose`` — if complex, generate focused sub-queries for each
         aspect.
      3. Caller answers each sub-query individually (RAGSystem handles this).
      4. ``synthesize`` — merge sub-answers into one coherent response.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._client = OpenAI(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.api_base_url,
        )
        self._model = cfg.llm.model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_complexity(self, query: str) -> Tuple[bool, List[str]]:
        """Detect whether *query* contains multiple independent aspects.

        Returns:
            (is_complex, aspects)  where aspects is a list of brief labels.
        """
        prompt = self_query_complexity_prompt(query)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            is_complex = bool(re.search(r"COMPLEX:\s*true", text, re.IGNORECASE))
            aspects: List[str] = []
            if "ASPECTS:" in text:
                section = text.split("ASPECTS:", 1)[1].strip()
                for line in section.splitlines():
                    line = line.strip().lstrip("-").strip()
                    if line:
                        aspects.append(line)
            return is_complex, aspects
        except Exception as exc:
            logger.error(f"analyze_complexity failed: {exc}")
            return False, [query]

    def decompose(self, query: str) -> SelfQueryDecomposition:
        """Return a :class:`SelfQueryDecomposition` for *query*.

        If the query is simple (single aspect), the decomposition contains
        one sub-query equal to the original query and ``is_complex=False``.
        """
        is_complex, aspects = self.analyze_complexity(query)

        if not is_complex or len(aspects) <= 1:
            logger.info("Query is simple — no decomposition needed.")
            return SelfQueryDecomposition(
                original_query=query,
                is_complex=False,
                sub_queries=[SubQuery(query=query, aspect="main")],
            )

        prompt = self_query_decompose_prompt(query, aspects)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            text = resp.choices[0].message.content.strip()
            sub_queries: List[SubQuery] = []
            for i, aspect in enumerate(aspects):
                match = re.search(
                    rf"Q{i + 1}:\s*(.+?)(?=Q{i + 2}:|$)", text, re.DOTALL | re.IGNORECASE
                )
                if match:
                    q = match.group(1).strip()
                    sub_queries.append(SubQuery(query=q, aspect=aspect))
                else:
                    # fallback: use the aspect label as a question
                    sub_queries.append(SubQuery(query=f"Tell me about {aspect}", aspect=aspect))

            if not sub_queries:
                raise ValueError("No sub-queries parsed from LLM response")

            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return SelfQueryDecomposition(
                original_query=query,
                is_complex=True,
                sub_queries=sub_queries,
            )
        except Exception as exc:
            logger.error(f"decompose failed: {exc}")
            return SelfQueryDecomposition(
                original_query=query,
                is_complex=False,
                sub_queries=[SubQuery(query=query, aspect="main")],
            )

    def synthesize(self, decomposition: SelfQueryDecomposition) -> str:
        """Merge all sub-query answers into one coherent response.

        Stores result in ``decomposition.synthesized_answer`` and also
        returns it as a string.
        """
        if not decomposition.is_complex:
            answer = decomposition.sub_queries[0].answer or ""
            decomposition.synthesized_answer = answer
            return answer

        parts = [
            f"**{sq.aspect.capitalize()}**\n{sq.answer}"
            for sq in decomposition.sub_queries
            if sq.answer
        ]
        if not parts:
            decomposition.synthesized_answer = ""
            return ""

        prompt = self_query_synthesize_prompt(
            decomposition.original_query, parts
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=900,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"synthesize failed: {exc}")
            answer = "\n\n".join(parts)

        decomposition.synthesized_answer = answer
        return answer
