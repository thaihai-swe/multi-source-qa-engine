"""Hallucination Detection & Mitigation for RAG responses.

Improves on the existing FactChecker by:
  1. Grounding *every* claim in the answer against the source documents with
     an explicit 0-10 confidence score.
  2. Computing an overall hallucination risk level (LOW / MEDIUM / HIGH).
  3. Automatically rewriting the answer to remove unsupported claims when the
     risk is MEDIUM or HIGH (auto-mitigation).
  4. Returning a fully transparent :class:`HallucinationReport` that the CLI
     can display so users understand exactly what changed and why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from openai import OpenAI

from src.config import get_config
from src.prompts.evaluation import (
    hallucination_grounding_prompt,
    hallucination_mitigation_prompt,
)
from src.utils import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ClaimGrounding:
    """Grounding result for a single claim extracted from the answer."""
    claim: str
    grounding_score: float           # 0-1; 1 = fully supported by sources
    best_source: Optional[str] = None
    supporting_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "grounding_score": round(self.grounding_score, 4),
            "best_source": self.best_source,
            "supporting_snippet": self.supporting_snippet,
        }


@dataclass
class HallucinationReport:
    """Full hallucination analysis for one RAG response.

    Attributes:
        overall_risk_level: "LOW", "MEDIUM", or "HIGH".
        grounding_score:    Average grounding across all claims (0-1).
        hallucination_score: 1 - grounding_score; for quick thresholding.
        unsupported_claims: Claims whose grounding_score < 0.5.
        supported_claims:   Claims whose grounding_score >= 0.5.
        mitigation_applied: Whether a refined answer was generated.
        refined_answer:     Rewritten answer (only when mitigation_applied).
        mitigation_explanation: Human-readable summary of what was changed.
    """
    query: str
    original_answer: str
    overall_risk_level: str
    grounding_score: float
    hallucination_score: float
    unsupported_claims: List[str] = field(default_factory=list)
    supported_claims: List[ClaimGrounding] = field(default_factory=list)
    mitigation_applied: bool = False
    refined_answer: Optional[str] = None
    mitigation_explanation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def final_answer(self) -> str:
        """Return refined_answer if mitigation was applied, else original."""
        return self.refined_answer if self.mitigation_applied and self.refined_answer else self.original_answer

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "overall_risk_level": self.overall_risk_level,
            "grounding_score": round(self.grounding_score, 4),
            "hallucination_score": round(self.hallucination_score, 4),
            "unsupported_claims": self.unsupported_claims,
            "supported_claims": [c.to_dict() for c in self.supported_claims],
            "mitigation_applied": self.mitigation_applied,
            "refined_answer": self.refined_answer,
            "mitigation_explanation": self.mitigation_explanation,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Claim-level hallucination detection with automatic mitigation.

    Risk thresholds
    ---------------
    - hallucination_score < 0.25  →  LOW   (answer is well-grounded)
    - 0.25 ≤ score < 0.50         →  MEDIUM (some unsupported claims)
    - score ≥ 0.50                 →  HIGH   (majority ungrounded)

    When *auto_mitigate=True* and risk is MEDIUM/HIGH, the detector asks the
    LLM to rewrite the answer removing or qualifying unsupported claims.
    """

    _HIGH_THRESHOLD = 0.50
    _MEDIUM_THRESHOLD = 0.25
    _MAX_CLAIMS = 8     # cap for cost control
    _MAX_SOURCES = 3    # top-N source docs used for grounding

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

    def analyze(
        self,
        query: str,
        answer: str,
        source_docs,  # List[RetrievedDocument] — avoid circular import
        auto_mitigate: bool = True,
    ) -> HallucinationReport:
        """Run hallucination analysis and optionally mitigate.

        Args:
            query:          The original user query.
            answer:         The LLM-generated answer to be checked.
            source_docs:    Retrieved :class:`RetrievedDocument` objects.
            auto_mitigate:  If True, rewrite answer when risk >= MEDIUM.

        Returns:
            A :class:`HallucinationReport` with full grounding details.
        """
        claims = self._extract_claims(answer)
        if not claims:
            return HallucinationReport(
                query=query,
                original_answer=answer,
                overall_risk_level="LOW",
                grounding_score=1.0,
                hallucination_score=0.0,
            )

        # Ground each claim
        groundings: List[ClaimGrounding] = [
            self._ground_claim(c, source_docs) for c in claims
        ]

        supported = [g for g in groundings if g.grounding_score >= 0.5]
        unsupported = [g for g in groundings if g.grounding_score < 0.5]

        avg_grounding = sum(g.grounding_score for g in groundings) / len(groundings)
        hallucination_score = 1.0 - avg_grounding
        risk_level = self._risk_level(hallucination_score)

        unsupported_claims = [g.claim for g in unsupported]
        refined_answer = None
        mitigation_explanation = None
        mitigation_applied = False

        if auto_mitigate and unsupported_claims and risk_level in ("MEDIUM", "HIGH"):
            refined_answer, mitigation_explanation = self._mitigate(
                query, answer, source_docs, unsupported_claims
            )
            mitigation_applied = True
            logger.info(f"Hallucination mitigation applied — {mitigation_explanation}")

        logger.info(
            f"HallucinationDetector: risk={risk_level} "
            f"grounding={avg_grounding:.2f} "
            f"unsupported={len(unsupported_claims)}/{len(claims)}"
        )

        return HallucinationReport(
            query=query,
            original_answer=answer,
            overall_risk_level=risk_level,
            grounding_score=avg_grounding,
            hallucination_score=hallucination_score,
            unsupported_claims=unsupported_claims,
            supported_claims=supported,
            mitigation_applied=mitigation_applied,
            refined_answer=refined_answer,
            mitigation_explanation=mitigation_explanation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_claims(self, text: str) -> List[str]:
        """Split *text* into individual verifiable sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20][: self._MAX_CLAIMS]

    def _ground_claim(self, claim: str, source_docs) -> ClaimGrounding:
        """Score how well *claim* is supported by *source_docs* (0-10 → 0-1)."""
        if not source_docs:
            return ClaimGrounding(claim=claim, grounding_score=0.0)

        context_parts = [
            f"[Source {i + 1} — {doc.source}]\n{doc.content[:400]}"
            for i, doc in enumerate(source_docs[: self._MAX_SOURCES])
        ]
        context = "\n---\n".join(context_parts)

        prompt = hallucination_grounding_prompt(claim, context)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            text = resp.choices[0].message.content.strip()

            score_match = re.search(r"SCORE:\s*(\d+)", text)
            raw_score = int(score_match.group(1)) if score_match else 5
            grounding = min(raw_score / 10.0, 1.0)

            evidence: Optional[str] = None
            ev_match = re.search(r"EVIDENCE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if ev_match and "none" not in ev_match.group(1).lower():
                evidence = ev_match.group(1).strip().strip('"')

            best_source: Optional[str] = None
            src_match = re.search(r"SOURCE:\s*(\d+)", text)
            if src_match:
                idx = int(src_match.group(1)) - 1
                if 0 <= idx < len(source_docs):
                    best_source = source_docs[idx].source

            return ClaimGrounding(
                claim=claim,
                grounding_score=grounding,
                best_source=best_source,
                supporting_snippet=evidence,
            )
        except Exception as exc:
            logger.error(f"_ground_claim failed: {exc}")
            return ClaimGrounding(claim=claim, grounding_score=0.5)

    def _risk_level(self, hallucination_score: float) -> str:
        if hallucination_score >= self._HIGH_THRESHOLD:
            return "HIGH"
        if hallucination_score >= self._MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    def _mitigate(
        self,
        query: str,
        answer: str,
        source_docs,
        unsupported_claims: List[str],
    ) -> Tuple[str, str]:
        """Rewrite *answer* to remove / qualify unsupported claims."""
        context = "\n---\n".join(
            f"[Source {i + 1} — {doc.source}]\n{doc.content[:500]}"
            for i, doc in enumerate(source_docs[: self._MAX_SOURCES])
        )

        prompt = hallucination_mitigation_prompt(
            query, answer, unsupported_claims, context
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=900,
            )
            refined = resp.choices[0].message.content.strip()
            count = len(unsupported_claims)
            preview = "; ".join(unsupported_claims[:2])
            explanation = (
                f"Removed/qualified {count} unsupported claim(s): {preview}"
                + (" …" if count > 2 else "")
            )
            return refined, explanation
        except Exception as exc:
            logger.error(f"_mitigate failed: {exc}")
            return answer, "Mitigation attempted but failed — original answer preserved."
