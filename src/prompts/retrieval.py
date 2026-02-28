"""Prompts used by the retrieval layer (domain guard)."""

from typing import List, Optional


# ===================================================================
# Domain Guard — topic extraction
# ===================================================================

def domain_topic_extraction_prompt(
    text_sample: str,
    source_label: Optional[str] = None,
) -> str:
    """Build the prompt for extracting main topics from document chunks."""
    label_hint = f"Source: {source_label}\n" if source_label else ""
    return (
        f"{label_hint}"
        f"Text sample:\n{text_sample}\n\n"
        "List 3-5 main topics covered by this text. "
        "Reply with ONLY a comma-separated list of short topic names, nothing else."
    )


# ===================================================================
# Domain Guard — query relevance scoring
# ===================================================================

def domain_relevance_prompt(
    query: str,
    topics: List[str],
    sample_preview: str,
) -> str:
    """Build the prompt for scoring query relevance to the loaded domain."""
    topics_str = ", ".join(topics) if topics else "unknown"
    return (
        "Rate how relevant the following query is to the given knowledge domain "
        "on a scale of 0 to 10.\n\n"
        f"Domain topics: {topics_str}\n"
        f"Sample domain content: {sample_preview}\n\n"
        f'Query: "{query}"\n\n'
        "Rules:\n"
        "  0  = completely unrelated to the domain\n"
        "  5  = tangentially related\n"
        "  10 = directly answered by the domain\n\n"
        "Respond with a SINGLE integer from 0 to 10. No explanation."
    )
