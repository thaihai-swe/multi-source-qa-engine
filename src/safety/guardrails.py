"""Guardrails and safety checks for RAG system"""
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from src.utils import get_logger

logger = get_logger()


@dataclass
class GuardrailResult:
    """Result from a guardrail check"""
    passed: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    message: str
    details: Optional[Dict] = None


class Guardrail(ABC):
    """Base class for all guardrails"""

    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        """Check if text passes the guardrail"""
        pass


class PromptInjectionDetector(Guardrail):
    """Detects potential prompt injection attacks"""

    def __init__(self):
        # Patterns that might indicate prompt injection attempts
        self.suspicious_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions?",
            r"disregard\s+(previous|above|all)",
            r"forget\s+(everything|all|previous)",
            r"new\s+instructions?:",
            r"system\s*:\s*you\s+are",
            r"</?\s*(system|instruction|prompt)\s*>",
            r"\\n\\n###\s+instruction",
            r"IGNORE\s+ALL",
            r"OVERRIDE\s+SYSTEM",
        ]
        self.pattern = re.compile("|".join(self.suspicious_patterns), re.IGNORECASE)

    def check(self, text: str) -> GuardrailResult:
        """Check for prompt injection patterns"""
        matches = self.pattern.findall(text)

        if matches:
            return GuardrailResult(
                passed=False,
                risk_level="HIGH",
                message="Potential prompt injection detected",
                details={"matches": matches}
            )

        # Check for excessive newlines or special tokens
        if text.count("\n") > 20 or "###" in text or "<|" in text:
            return GuardrailResult(
                passed=True,
                risk_level="MEDIUM",
                message="Suspicious formatting detected",
                details={"newlines": text.count("\n")}
            )

        return GuardrailResult(
            passed=True,
            risk_level="LOW",
            message="No injection patterns detected",
            details=None
        )


class PIIDetector(Guardrail):
    """Detects Personally Identifiable Information"""

    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }

    def check(self, text: str) -> GuardrailResult:
        """Check for PII in text"""
        found_pii = {}

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                found_pii[pii_type] = len(matches)

        if found_pii:
            risk_level = "HIGH" if any(k in found_pii for k in ["ssn", "credit_card"]) else "MEDIUM"
            return GuardrailResult(
                passed=False,
                risk_level=risk_level,
                message=f"PII detected: {', '.join(found_pii.keys())}",
                details=found_pii
            )

        return GuardrailResult(
            passed=True,
            risk_level="LOW",
            message="No PII detected",
            details=None
        )

    @staticmethod
    def redact_pii(text: str) -> str:
        """Redact PII from text"""
        # Simple redaction - replace with placeholder
        patterns = {
            "email": (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "[EMAIL]"),
            "phone": (re.compile(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'), "[PHONE]"),
            "ssn": (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN]"),
            "credit_card": (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), "[CREDIT_CARD]"),
        }

        redacted = text
        for pii_type, (pattern, placeholder) in patterns.items():
            redacted = pattern.sub(placeholder, redacted)

        return redacted


class ToxicityFilter(Guardrail):
    """Filters toxic or inappropriate content"""

    def __init__(self):
        # Simple keyword-based toxicity detection
        # In production, use a proper toxicity classifier model
        self.toxic_keywords = {
            "profanity": ["fuck", "shit", "damn", "bastard", "asshole"],
            "hate_speech": ["nazi", "terrorist", "extremist"],
            "violence": ["kill", "murder", "attack", "destroy"],
        }

    def check(self, text: str) -> GuardrailResult:
        """Check for toxic content"""
        text_lower = text.lower()
        found_issues = defaultdict(list)

        for category, keywords in self.toxic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_issues[category].append(keyword)

        if found_issues:
            return GuardrailResult(
                passed=False,
                risk_level="MEDIUM",
                message="Potentially toxic content detected",
                details=dict(found_issues)
            )

        return GuardrailResult(
            passed=True,
            risk_level="LOW",
            message="No toxic content detected",
            details=None
        )


class RateLimiter:
    """Rate limiting for API calls"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)  # user_id -> deque of timestamps

    def check_rate_limit(self, user_id: str = "default") -> Tuple[bool, str]:
        """
        Check if request is within rate limit

        Returns:
            (allowed, message)
        """
        now = time.time()
        user_requests = self.requests[user_id]

        # Remove old requests outside the window
        while user_requests and user_requests[0] < now - self.window_seconds:
            user_requests.popleft()

        # Check if under limit
        if len(user_requests) >= self.max_requests:
            wait_time = int(user_requests[0] + self.window_seconds - now)
            return False, f"Rate limit exceeded. Try again in {wait_time} seconds."

        # Add current request
        user_requests.append(now)
        remaining = self.max_requests - len(user_requests)

        return True, f"Rate limit: {remaining} requests remaining"

    def reset(self, user_id: str = "default"):
        """Reset rate limit for a user"""
        if user_id in self.requests:
            del self.requests[user_id]


class InputGuardrail:
    """Combines multiple input guardrails"""

    def __init__(
        self,
        check_injection: bool = True,
        check_pii: bool = True,
        check_toxicity: bool = True,
        max_length: int = 5000
    ):
        self.check_injection = check_injection
        self.check_pii = check_pii
        self.check_toxicity = check_toxicity
        self.max_length = max_length

        self.injection_detector = PromptInjectionDetector() if check_injection else None
        self.pii_detector = PIIDetector() if check_pii else None
        self.toxicity_filter = ToxicityFilter() if check_toxicity else None

    def validate(self, text: str) -> Tuple[bool, List[GuardrailResult]]:
        """
        Validate input text against all guardrails

        Returns:
            (is_valid, list of guardrail results)
        """
        results = []

        # Check length
        if len(text) > self.max_length:
            results.append(GuardrailResult(
                passed=False,
                risk_level="MEDIUM",
                message=f"Input too long ({len(text)} > {self.max_length} chars)",
                details={"length": len(text), "max": self.max_length}
            ))
            return False, results

        # Check for empty input
        if not text.strip():
            results.append(GuardrailResult(
                passed=False,
                risk_level="LOW",
                message="Empty input",
                details=None
            ))
            return False, results

        # Run all checks
        if self.injection_detector:
            result = self.injection_detector.check(text)
            results.append(result)
            if not result.passed:
                return False, results

        if self.pii_detector:
            result = self.pii_detector.check(text)
            results.append(result)
            # PII is a warning, not a blocker

        if self.toxicity_filter:
            result = self.toxicity_filter.check(text)
            results.append(result)
            if not result.passed:
                return False, results

        # All checks passed
        return True, results


class OutputGuardrail:
    """Guardrails for generated output"""

    def __init__(
        self,
        check_pii: bool = True,
        min_length: int = 10,
        max_length: int = 10000
    ):
        self.check_pii = check_pii
        self.min_length = min_length
        self.max_length = max_length

        self.pii_detector = PIIDetector() if check_pii else None

    def validate(self, text: str, auto_redact: bool = False) -> Tuple[bool, str, List[GuardrailResult]]:
        """
        Validate output text

        Args:
            text: Output text to validate
            auto_redact: If True, automatically redact PII

        Returns:
            (is_valid, processed_text, list of guardrail results)
        """
        results = []
        processed_text = text

        # Check length
        if len(text) < self.min_length:
            results.append(GuardrailResult(
                passed=False,
                risk_level="LOW",
                message=f"Output too short ({len(text)} < {self.min_length} chars)",
                details={"length": len(text)}
            ))
            return False, processed_text, results

        if len(text) > self.max_length:
            results.append(GuardrailResult(
                passed=False,
                risk_level="MEDIUM",
                message=f"Output too long ({len(text)} > {self.max_length} chars)",
                details={"length": len(text)}
            ))
            return False, processed_text, results

        # Check for PII
        if self.pii_detector:
            result = self.pii_detector.check(text)
            results.append(result)

            if not result.passed and auto_redact:
                processed_text = PIIDetector.redact_pii(text)
                logger.warning("⚠️ PII detected and redacted from output")

        return True, processed_text, results


__all__ = [
    "Guardrail",
    "GuardrailResult",
    "PromptInjectionDetector",
    "PIIDetector",
    "ToxicityFilter",
    "RateLimiter",
    "InputGuardrail",
    "OutputGuardrail"
]
