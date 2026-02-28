"""Passage highlighting and relevance extraction for source attribution"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re
from nltk.tokenize import sent_tokenize
import numpy as np
from src.models import RetrievedDocument
from src.utils import get_logger

logger = get_logger()


@dataclass
class HighlightedPassage:
    """Represents a highlighted passage from a source document"""
    text: str
    document_source: str
    document_index: int
    relevance_score: float  # 0-1, higher is more relevant
    start_pos: int  # Position in original document
    end_pos: int
    sentence_index: int  # Which sentence in the document

    def to_dict(self):
        return {
            "text": self.text,
            "source": self.document_source,
            "relevance_score": self.relevance_score,
            "sentence_index": self.sentence_index
        }


class PassageHighlighter:
    """Extract and highlight the most relevant passages from retrieved documents"""

    def __init__(self, max_passages_per_doc: int = 3):
        """
        Args:
            max_passages_per_doc: Maximum number of passages to extract per document
        """
        self.max_passages_per_doc = max_passages_per_doc

    def extract_relevant_passages(
        self,
        query: str,
        documents: List[RetrievedDocument],
        answer: str = None
    ) -> List[HighlightedPassage]:
        """
        Extract the most relevant passages from retrieved documents.

        Args:
            query: User's query
            documents: Retrieved documents
            answer: Generated answer (optional, used for additional relevance scoring)

        Returns:
            List of highlighted passages sorted by relevance
        """
        all_passages = []

        # Extract query keywords
        query_keywords = self._extract_keywords(query)
        answer_keywords = self._extract_keywords(answer) if answer else set()

        for doc in documents:
            passages = self._extract_passages_from_document(
                doc, query_keywords, answer_keywords
            )
            all_passages.extend(passages)

        # Sort by relevance and return top passages
        all_passages.sort(key=lambda p: p.relevance_score, reverse=True)

        # Limit total number of passages
        max_total = len(documents) * self.max_passages_per_doc
        return all_passages[:max_total]

    def _extract_passages_from_document(
        self,
        document: RetrievedDocument,
        query_keywords: set,
        answer_keywords: set
    ) -> List[HighlightedPassage]:
        """Extract top passages from a single document"""
        content = document.content

        # Split into sentences
        try:
            sentences = sent_tokenize(content)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}, using fallback")
            sentences = content.split('. ')

        passages = []
        current_pos = 0

        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Calculate relevance score for this sentence
            relevance = self._calculate_sentence_relevance(
                sentence, query_keywords, answer_keywords
            )

            # Add document retrieval score as a base relevance boost
            doc_relevance = document.relevance_score if hasattr(document, 'relevance_score') else 0.5
            combined_relevance = (relevance * 0.7) + (doc_relevance * 0.3)

            start_pos = current_pos
            end_pos = current_pos + len(sentence)

            passage = HighlightedPassage(
                text=sentence.strip(),
                document_source=document.source,
                document_index=document.index,
                relevance_score=combined_relevance,
                start_pos=start_pos,
                end_pos=end_pos,
                sentence_index=idx
            )
            passages.append(passage)

            current_pos = end_pos + 1  # +1 for the space/period

        # Return top N passages from this document
        passages.sort(key=lambda p: p.relevance_score, reverse=True)
        return passages[:self.max_passages_per_doc]

    def _calculate_sentence_relevance(
        self,
        sentence: str,
        query_keywords: set,
        answer_keywords: set
    ) -> float:
        """
        Calculate relevance score for a sentence based on keyword overlap.

        Returns score between 0 and 1.
        """
        sentence_lower = sentence.lower()
        sentence_words = set(self._tokenize(sentence_lower))

        # Count keyword matches
        query_matches = len(query_keywords & sentence_words)
        answer_matches = len(answer_keywords & sentence_words)

        # Calculate overlap ratios
        query_ratio = query_matches / len(query_keywords) if query_keywords else 0
        answer_ratio = answer_matches / len(answer_keywords) if answer_keywords else 0

        # Weighted combination (query matches are more important)
        relevance = (query_ratio * 0.6) + (answer_ratio * 0.4)

        # Boost for exact phrase matches
        for keyword in query_keywords:
            if keyword in sentence_lower and len(keyword) > 4:  # Multi-character keywords
                relevance += 0.1

        return min(relevance, 1.0)  # Cap at 1.0

    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        if not text:
            return set()

        # Common stopwords to filter
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
            'how', 'why', 'with', 'as', 'by', 'from'
        }

        words = self._tokenize(text.lower())
        keywords = {
            word for word in words
            if word not in stopwords and len(word) > 2
        }

        return keywords

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def format_highlighted_passages(
        self,
        passages: List[HighlightedPassage],
        max_display: int = 5
    ) -> str:
        """Format highlighted passages for display"""
        if not passages:
            return "No highlighted passages available."

        output = []
        output.append("\n" + "="*80)
        output.append("📍 HIGHLIGHTED PASSAGES (Top Relevant Excerpts)")
        output.append("="*80)

        for i, passage in enumerate(passages[:max_display], 1):
            output.append(f"\n[Passage {i}] Relevance: {passage.relevance_score:.2%}")
            output.append(f"Source: {passage.document_source}")
            output.append(f"📝 \"{passage.text}\"")
            if i < min(len(passages), max_display):
                output.append("-" * 80)

        if len(passages) > max_display:
            output.append(f"\n... and {len(passages) - max_display} more passages")

        return "\n".join(output)


__all__ = ["PassageHighlighter", "HighlightedPassage"]
