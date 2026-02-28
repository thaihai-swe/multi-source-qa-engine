"""Domain Guard — semantic similarity thresholding for out-of-domain detection.

Builds a lightweight topic profile from loaded document chunks and uses LLM-based
relevance scoring to warn users when their query falls outside the knowledge domain.

Why this matters
----------------
Without domain awareness a RAG system silently retrieves the "least bad" chunks
and generates an answer that sounds plausible but is actually hallucinated.
DomainGuard surfaces this risk *before* generation so the user can load better
sources or rephrase their question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import get_config
from src.prompts.retrieval import (
    domain_topic_extraction_prompt,
    domain_relevance_prompt,
)
from src.utils import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DomainCheckResult:
    """Result of a domain-relevance check for a single query."""
    query: str
    is_in_domain: bool
    similarity_score: float          # 0-1; higher = more relevant to domain
    domain_topics: List[str]
    warning_message: Optional[str] = None
    recommendation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "is_in_domain": self.is_in_domain,
            "similarity_score": round(self.similarity_score, 4),
            "domain_topics": self.domain_topics,
            "warning_message": self.warning_message,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

class DomainGuard:
    """Detects out-of-domain queries by comparing them against a domain profile.

    Usage (typical RAGSystem integration):

        guard = DomainGuard(threshold=0.35)

        # After loading documents:
        guard.build_domain_profile(chunks)

        # Before every query:
        result = guard.check_domain_relevance(query)
        if not result.is_in_domain:
            print(result.warning_message)

    The domain profile is purely text-based (up to 50 sample chunks + extracted
    topics), so it works even when a dedicated embedding endpoint is unavailable.
    """

    _DEFAULT_THRESHOLD = 0.35

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        cfg = get_config()
        self._client = OpenAI(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.api_base_url,
        )
        self._model = cfg.llm.model_name
        self.threshold = threshold
        self.domain_topics: List[str] = []
        self._sample_chunks: List[str] = []
        self._has_profile: bool = False

    # ------------------------------------------------------------------
    # Profile building
    # ------------------------------------------------------------------

    def build_domain_profile(
        self,
        documents: List[str],
        source_label: Optional[str] = None,
    ) -> None:
        """Build (or extend) the domain profile from *documents*.

        Args:
            documents:    List of text chunks from the loaded source.
            source_label: Optional human-readable label (e.g. the source URL /
                          Wikipedia topic) used to improve topic naming.
        """
        if not documents:
            logger.warning("DomainGuard: no documents provided — profile unchanged.")
            return

        # Keep at most 50 representative chunks (first + every N-th)
        step = max(1, len(documents) // 50)
        sampled = documents[::step][:50]
        self._sample_chunks.extend(sampled)
        self._sample_chunks = self._sample_chunks[-50:]  # rolling window

        new_topics = self._extract_topics(sampled[:8], source_label)
        # Merge without duplicates (case-insensitive)
        existing_lower = {t.lower() for t in self.domain_topics}
        for t in new_topics:
            if t.lower() not in existing_lower:
                self.domain_topics.append(t)
                existing_lower.add(t.lower())

        self._has_profile = True
        logger.info(
            f"DomainGuard: profile updated — "
            f"{len(self._sample_chunks)} chunks, topics: {self.domain_topics}"
        )

    def reset(self) -> None:
        """Clear the current domain profile."""
        self.domain_topics = []
        self._sample_chunks = []
        self._has_profile = False

    # ------------------------------------------------------------------
    # Relevance checking
    # ------------------------------------------------------------------

    def check_domain_relevance(self, query: str) -> DomainCheckResult:
        """Return a :class:`DomainCheckResult` for *query*.

        When no profile has been built yet the query is treated as in-domain
        (no false alarms before the first ``load`` command).
        """
        if not self._has_profile:
            return DomainCheckResult(
                query=query,
                is_in_domain=True,
                similarity_score=1.0,
                domain_topics=[],
            )

        score = self._llm_relevance_score(query)
        in_domain = score >= self.threshold

        warning: Optional[str] = None
        recommendation: Optional[str] = None
        if not in_domain:
            topics_str = (
                ", ".join(self.domain_topics)
                if self.domain_topics
                else "the currently loaded documents"
            )
            warning = (
                f"Query appears to be outside the loaded knowledge domain "
                f"(relevance score: {score:.0%}). "
                "The answer may be unreliable or hallucinated."
            )
            recommendation = (
                f"The loaded knowledge base covers: {topics_str}. "
                "Consider loading a relevant source with the 'load' command "
                "before querying this topic."
            )

        logger.info(
            f"DomainGuard: query='{query[:60]}' "
            f"score={score:.3f} in_domain={in_domain}"
        )
        return DomainCheckResult(
            query=query,
            is_in_domain=in_domain,
            similarity_score=score,
            domain_topics=list(self.domain_topics),
            warning_message=warning,
            recommendation=recommendation,
        )

    def get_domain_stats(self) -> Dict:
        """Return a dictionary with current profile statistics."""
        return {
            "has_profile": self._has_profile,
            "sample_chunks_count": len(self._sample_chunks),
            "domain_topics": list(self.domain_topics),
            "similarity_threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_topics(
        self, chunks: List[str], source_label: Optional[str]
    ) -> List[str]:
        """Ask the LLM to extract 3-5 main topics from *chunks*."""
        combined = " ".join(chunks)[:2000]
        prompt = domain_topic_extraction_prompt(combined, source_label)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            topics = [t.strip() for t in raw.split(",") if t.strip()]
            return topics[:5]
        except Exception as exc:
            logger.error(f"DomainGuard._extract_topics failed: {exc}")
            return ["general knowledge"]

    def _llm_relevance_score(self, query: str) -> float:
        """Use the LLM to score query relevance to the domain (0-10 → 0-1)."""
        sample_preview = self._sample_chunks[0][:400] if self._sample_chunks else ""

        prompt = domain_relevance_prompt(
            query, self.domain_topics, sample_preview
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            numbers = re.findall(r"\d+(?:\.\d+)?", raw)
            if numbers:
                return min(float(numbers[0]) / 10.0, 1.0)
        except Exception as exc:
            logger.error(f"DomainGuard._llm_relevance_score failed: {exc}")
        return 1.0  # fail-open: assume in-domain on error
