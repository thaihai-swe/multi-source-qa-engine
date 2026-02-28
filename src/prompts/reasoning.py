"""Prompts used by reasoning modules (query expansion, multi-hop, self-query)."""

from typing import List


# ===================================================================
# Query Expander
# ===================================================================

QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You are a query optimization expert. Generate alternative query formulations."
)


def query_expansion_user_prompt(query: str, num_variations: int) -> str:
    """Build the user message that asks for query variations."""
    return (
        f"Generate {num_variations} alternative phrasings and perspectives for this query.\n"
        "Each variation should ask the same thing but from different angles or with different wording.\n"
        "Make them diverse: paraphrasings, synonyms, decompositions, and related questions.\n\n"
        f"Original Query: {query}\n\n"
        "Return ONLY the variations, one per line, without numbering or extra formatting."
    )


# ===================================================================
# Multi-Hop Reasoner — decompose
# ===================================================================

MULTI_HOP_DECOMPOSE_SYSTEM_PROMPT = (
    "You are an expert at decomposing complex questions into simpler steps."
)


def multi_hop_decompose_user_prompt(query: str, max_steps: int) -> str:
    """Build the user message for breaking a complex query into sub-questions."""
    return (
        f"Break down this complex query into {max_steps} simpler sub-questions that together help answer it.\n"
        "Each sub-question should build on previous understanding.\n\n"
        f"Complex Query: {query}\n\n"
        "Return ONLY the sub-questions, one per line, without numbering or extra formatting.\n"
        "Each should be a complete, standalone question."
    )


# ===================================================================
# Multi-Hop Reasoner — synthesize
# ===================================================================

MULTI_HOP_SYNTHESIZE_SYSTEM_PROMPT = (
    "You are synthesizing multi-hop reasoning into a comprehensive answer."
)


def multi_hop_synthesize_user_prompt(query: str, step_summary: str) -> str:
    """Build the user message for synthesizing step-by-step results."""
    return (
        "Based on the following step-by-step reasoning, provide a comprehensive answer to the original query.\n"
        "Synthesize all the information into a coherent, unified response.\n\n"
        f"Original Query: {query}\n\n"
        "Step-by-step Results:\n"
        f"{step_summary}\n\n"
        "Provide a synthesized answer that incorporates all findings."
    )


# ===================================================================
# Self-Query Decomposer — complexity analysis
# ===================================================================

def self_query_complexity_prompt(query: str) -> str:
    """Build the prompt that determines whether a query has multiple aspects."""
    return (
        "Analyze whether the following query asks about multiple DISTINCT, "
        "independent topics that would benefit from separate answers.\n\n"
        f"Query: {query}\n\n"
        "A query is COMPLEX if it:\n"
        "  - Asks about 2 or more clearly separate topics\n"
        "  - Uses connectors like 'and', 'also', 'as well as', 'additionally'\n"
        "  - Contains multiple question marks\n\n"
        "A query is SIMPLE if it focuses on one topic (even if detailed).\n\n"
        "Reply in EXACTLY this format (no extra text):\n"
        "COMPLEX: true\n"
        "ASPECTS:\n"
        "- aspect one\n"
        "- aspect two\n\n"
        "OR\n\n"
        "COMPLEX: false\n"
        "ASPECTS:\n"
        "- main topic"
    )


# ===================================================================
# Self-Query Decomposer — sub-query generation
# ===================================================================

def self_query_decompose_prompt(query: str, aspects: List[str]) -> str:
    """Build the prompt that generates focused sub-questions per aspect."""
    aspect_lines = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(aspects))
    return (
        f"Break the following complex query into {len(aspects)} focused, "
        "independent sub-questions — one per aspect.\n\n"
        f"Original query: {query}\n\n"
        "Aspects identified:\n"
        f"{aspect_lines}\n\n"
        "Write a clear, self-contained question for each aspect.\n"
        "Use this format EXACTLY:\n"
        "Q1: <question for aspect 1>\n"
        "Q2: <question for aspect 2>\n"
        "(continue for all aspects)"
    )


# ===================================================================
# Self-Query Decomposer — synthesis
# ===================================================================

def self_query_synthesize_prompt(
    original_query: str, answer_parts: List[str]
) -> str:
    """Build the prompt that merges sub-answers into one coherent response."""
    joined = "\n\n".join(answer_parts)
    return (
        "You are merging answers to different aspects of a complex question "
        "into one well-structured response.\n\n"
        f"Original question: {original_query}\n\n"
        "Answers to individual aspects:\n\n"
        f"{joined}\n\n"
        "Instructions:\n"
        "- Write a single, comprehensive answer.\n"
        "- Use clear paragraph or section breaks between topics.\n"
        "- Avoid repetition.\n"
        "- Do NOT mention that these were separate sub-questions.\n"
        "Synthesized answer:"
    )
