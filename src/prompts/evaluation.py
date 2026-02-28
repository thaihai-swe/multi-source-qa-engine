"""Prompts used by evaluation modules (RAGAS, fact-checker, hallucination)."""

from typing import List


# ===================================================================
# RAGAS Evaluator — context relevance
# ===================================================================

RAGAS_CONTEXT_RELEVANCE_SYSTEM_PROMPT = (
    "You are evaluating the relevance of document context to a query. "
    "Respond with only a number from 0 to 10."
)


def ragas_context_relevance_user_prompt(query: str, context: str) -> str:
    """Build the user message for scoring context relevance."""
    return (
        f"Query: {query}\n\n"
        f"Context: {context[:500]}\n\n"
        "On a scale of 0-10, how relevant is this context to the query?\n"
        "Only provide a number."
    )


# ===================================================================
# RAGAS Evaluator — answer relevance
# ===================================================================

RAGAS_ANSWER_RELEVANCE_SYSTEM_PROMPT = (
    "You are evaluating if an answer directly addresses a query. "
    "Respond with only a number from 0 to 10."
)


def ragas_answer_relevance_user_prompt(query: str, answer: str) -> str:
    """Build the user message for scoring answer relevance."""
    return (
        f"Query: {query}\n\n"
        f"Answer: {answer[:500]}\n\n"
        "On a scale of 0-10, how well does this answer address the query?\n"
        "Only provide a number."
    )


# ===================================================================
# RAGAS Evaluator — faithfulness
# ===================================================================

RAGAS_FAITHFULNESS_SYSTEM_PROMPT = (
    "You are evaluating if an answer is grounded in provided context. "
    "Respond with only a number from 0 to 10."
)


def ragas_faithfulness_user_prompt(context: str, answer: str) -> str:
    """Build the user message for scoring faithfulness."""
    return (
        f"Context: {context[:500]}\n\n"
        f"Answer: {answer[:500]}\n\n"
        "On a scale of 0-10, how much of the answer is supported by the context?\n"
        "Only provide a number."
    )


# ===================================================================
# Fact Checker
# ===================================================================

def fact_check_prompt(fact: str, context: str) -> str:
    """Build the prompt for checking a statement against context."""
    return (
        "Based on the provided context, determine if the following statement "
        "is supported, contradicted, or unknown:\n\n"
        f'Statement: "{fact}"\n\n'
        f"Context:\n{context}\n\n"
        "Respond in this exact format:\n"
        "SUPPORTED|CONTRADICTED|UNKNOWN\n"
        "Confidence: [0-100]\n"
        "Evidence: [brief explanation]"
    )


# ===================================================================
# Hallucination Detector — claim grounding
# ===================================================================

def hallucination_grounding_prompt(claim: str, context: str) -> str:
    """Build the prompt for evaluating whether a claim is source-grounded."""
    return (
        "Evaluate if the following claim is supported by the source documents.\n\n"
        f'Claim: "{claim}"\n\n'
        f"Sources:\n{context}\n\n"
        "Reply in EXACTLY this format (no extra text):\n"
        "SCORE: <integer 0-10>\n"
        "EVIDENCE: <one-sentence quote from the sources, or 'none'>\n"
        "SOURCE: <source number, e.g. 1, or 'none'>"
    )


# ===================================================================
# Hallucination Detector — mitigation (answer rewrite)
# ===================================================================

def hallucination_mitigation_prompt(
    query: str,
    answer: str,
    unsupported_claims: List[str],
    context: str,
) -> str:
    """Build the prompt for rewriting an answer to remove hallucinated claims."""
    unsupported_text = "\n".join(f"  - {c}" for c in unsupported_claims)
    return (
        "You are refining an AI-generated answer to remove hallucinated claims.\n\n"
        f"Original question: {query}\n\n"
        f"Original answer:\n{answer}\n\n"
        "Claims NOT supported by the source documents (potential hallucinations):\n"
        f"{unsupported_text}\n\n"
        "Verified source documents:\n"
        f"{context}\n\n"
        "Write a revised answer that:\n"
        "1. Removes or qualifies every unsupported claim listed above.\n"
        "2. Preserves all well-supported information.\n"
        "3. Acknowledges uncertainty where needed "
        '(e.g., "Based on the available sources…").\n'
        "4. Maintains a helpful, natural tone.\n\n"
        "Revised answer:"
    )
