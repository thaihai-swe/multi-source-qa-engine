"""Retrieval layer - abstract base classes and concrete implementations"""
from src.retrieval.base import Retriever, DocumentLoader, Chunker
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.cache import EmbeddingCache
from src.retrieval.chunker import AdaptiveChunker
from src.retrieval.loader import MultiSourceDataLoader
from src.retrieval.domain_guard import DomainGuard, DomainCheckResult
from src.retrieval.passage_highlighter import PassageHighlighter, HighlightedPassage
from src.retrieval.reranker import DocumentReranker, RerankerConfig, HyDEGenerator

__all__ = [
    "Retriever",
    "DocumentLoader",
    "Chunker",
    "HybridSearchEngine",
    "EmbeddingCache",
    "AdaptiveChunker",
    "MultiSourceDataLoader",
    "DomainGuard",
    "DomainCheckResult",
    "PassageHighlighter",
    "HighlightedPassage",
    "DocumentReranker",
    "RerankerConfig",
    "HyDEGenerator",
]
