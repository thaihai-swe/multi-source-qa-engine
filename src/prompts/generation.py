"""Prompts used by the answer-generation layer."""

# ---------------------------------------------------------------------------
# LLM Answer Generator — buffered
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful assistant. Based on the provided context,\n"
    "answer the user's question accurately and concisely. If the context doesn't contain enough information,\n"
    "acknowledge this and provide your best response based on what is available."
)


def answer_user_prompt(context_text: str, query: str) -> str:
    """Build the user message for buffered answer generation."""
    return f"{context_text}\n\nQuestion: {query}"


# ---------------------------------------------------------------------------
# LLM Answer Generator — streaming
# ---------------------------------------------------------------------------

ANSWER_STREAMING_SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful assistant. Based on the provided context,\n"
    "answer the user's question accurately and concisely."
)


def answer_streaming_user_prompt(context_text: str, query: str) -> str:
    """Build the user message for streaming answer generation."""
    return f"{context_text}\n\nQuestion: {query}"
