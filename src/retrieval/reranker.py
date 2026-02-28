"""Advanced reranking and diversity optimization for retrieval"""
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from src.models import RetrievedDocument
from src.utils import get_logger

logger = get_logger()

# Try to import sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers not installed. Cross-encoder reranking disabled.")


@dataclass
class RerankerConfig:
    """Configuration for reranking"""
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    use_mmr: bool = True
    mmr_lambda: float = 0.7  # Balance between relevance (1.0) and diversity (0.0)
    max_results: int = 5


class DocumentReranker:
    """Rerank retrieved documents using cross-encoder and diversity optimization"""

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Args:
            config: Reranker configuration. If None, uses defaults.
        """
        self.config = config or RerankerConfig()
        self.cross_encoder = None

        # Initialize cross-encoder if available and enabled
        if CROSS_ENCODER_AVAILABLE and self.config.use_cross_encoder:
            try:
                logger.info(f"Loading cross-encoder model: {self.config.cross_encoder_model}")
                self.cross_encoder = CrossEncoder(self.config.cross_encoder_model)
                logger.info("✅ Cross-encoder loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load cross-encoder: {e}")
                self.cross_encoder = None

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder and/or MMR.

        Args:
            query: User's query
            documents: Retrieved documents to rerank
            top_k: Number of top documents to return (default: use config)

        Returns:
            Reranked documents
        """
        if not documents:
            return documents

        top_k = top_k or self.config.max_results

        # Step 1: Cross-encoder reranking (if available)
        if self.cross_encoder and self.config.use_cross_encoder:
            documents = self._cross_encoder_rerank(query, documents)
            logger.info(f"✅ Cross-encoder reranking applied")

        # Step 2: MMR for diversity (if enabled)
        if self.config.use_mmr:
            documents = self._mmr_rerank(query, documents, top_k)
            logger.info(f"✅ MMR diversity optimization applied (λ={self.config.mmr_lambda})")
        else:
            # Just return top k by current scores
            documents = documents[:top_k]

        return documents

    def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder model.

        Cross-encoders score query-document pairs more accurately than bi-encoders
        by jointly encoding query and document.
        """
        if not self.cross_encoder:
            return documents

        try:
            # Prepare query-document pairs
            pairs = [[query, doc.content] for doc in documents]

            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)

            # Update documents with new scores and rerank
            for i, doc in enumerate(documents):
                # Store original distance for reference
                if not hasattr(doc, 'original_distance'):
                    doc.original_distance = doc.distance

                # Convert cross-encoder score to distance (lower is better)
                # Cross-encoder scores are typically -10 to 10, normalize to 0-2 distance
                doc.distance = 1.0 - (scores[i] / 10.0)  # Higher score = lower distance
                doc.distance = max(0.0, min(2.0, doc.distance))  # Clamp to [0, 2]

            # Sort by new distance (lower is better)
            documents.sort(key=lambda d: d.distance)

            return documents

        except Exception as e:
            logger.error(f"❌ Cross-encoder reranking failed: {e}")
            return documents

    def _mmr_rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int
    ) -> List[RetrievedDocument]:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity.

        MMR balances relevance and diversity by penalizing documents
        that are too similar to already selected documents.

        Formula: MMR = λ * Relevance(q,d) - (1-λ) * max Similarity(d, selected)
        """
        if len(documents) <= top_k:
            return documents

        try:
            # Get relevance scores (convert distance to relevance)
            relevance_scores = np.array([
                max(0, 1 - (doc.distance / 2))
                for doc in documents
            ])

            # Initialize
            selected_indices = []
            remaining_indices = list(range(len(documents)))

            # Select first document (highest relevance)
            first_idx = np.argmax(relevance_scores)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)

            # Incrementally select documents that maximize MMR
            while len(selected_indices) < top_k and remaining_indices:
                mmr_scores = []

                for idx in remaining_indices:
                    # Relevance component
                    relevance = relevance_scores[idx]

                    # Diversity component (max similarity to selected documents)
                    max_similarity = max([
                        self._compute_similarity(
                            documents[idx].content,
                            documents[sel_idx].content
                        )
                        for sel_idx in selected_indices
                    ])

                    # MMR score
                    mmr = (self.config.mmr_lambda * relevance) - \
                          ((1 - self.config.mmr_lambda) * max_similarity)
                    mmr_scores.append(mmr)

                # Select document with highest MMR score
                best_mmr_idx = np.argmax(mmr_scores)
                best_doc_idx = remaining_indices[best_mmr_idx]
                selected_indices.append(best_doc_idx)
                remaining_indices.remove(best_doc_idx)

            # Return selected documents in order
            return [documents[idx] for idx in selected_indices]

        except Exception as e:
            logger.error(f"❌ MMR reranking failed: {e}")
            return documents[:top_k]

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using simple word overlap.

        For better results, use embeddings, but this is faster and sufficient
        for diversity filtering.
        """
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE) generator.

    HyDE improves retrieval by generating a hypothetical answer first,
    then using that answer to retrieve documents.
    """

    def __init__(self, llm_generator=None):
        """
        Args:
            llm_generator: LLM generator instance for creating hypothetical documents
        """
        self.llm_generator = llm_generator

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        This is used to improve retrieval by searching for documents
        similar to what the answer might look like.

        Args:
            query: User's query

        Returns:
            Hypothetical document text
        """
        if not self.llm_generator:
            logger.warning("⚠️ No LLM generator provided for HyDE")
            return query

        try:
            # Create prompt for hypothetical document generation
            prompt = f"""Generate a brief, factual paragraph that would answer this question:

Question: {query}

Answer (2-3 sentences, factual style):"""

            # Generate using LLM (simplified - actual implementation would use the generator)
            # For now, just return the query as fallback
            logger.info("HyDE generation not yet fully implemented")
            return query

        except Exception as e:
            logger.error(f"❌ HyDE generation failed: {e}")
            return query


__all__ = ["DocumentReranker", "RerankerConfig", "HyDEGenerator"]
