"""Smart chunking strategy that automatically determines optimal chunk sizes"""
import re
from typing import Dict, Tuple, List, Optional
from src.utils import get_logger

logger = get_logger()


class SmartChunkSizer:
    """Analyzes document characteristics and recommends optimal chunk sizes"""

    def __init__(self):
        # Preset configurations for different document types
        self.presets = {
            "wikipedia": {"child": 256, "parent": 1024},
            "academic": {"child": 400, "parent": 1600},
            "technical_docs": {"child": 300, "parent": 1200},
            "blog": {"child": 200, "parent": 800},
            "code_docs": {"child": 180, "parent": 720},
            "fiction": {"child": 350, "parent": 1400},
            "news": {"child": 200, "parent": 800},
        }

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple word-to-token conversion"""
        # Average: 1 token ≈ 4 characters or ~0.75 words
        words = len(text.split())
        # Use words * 1.3 as a conservative estimate (most texts average 1.3 tokens/word)
        return max(1, int(words * 1.3))

    def analyze_document(self, text: str) -> Dict:
        """Analyze document characteristics"""
        analysis = {
            "total_length": len(text),
            "token_estimate": self.estimate_tokens(text),
            "line_count": len(text.split('\n')),
            "content_type": self._detect_content_type(text),
            "domain": self._detect_domain(text),
            "complexity_score": self._calculate_complexity(text),
            "structure_score": self._analyze_structure(text),
        }
        return analysis

    def recommend_chunk_sizes(self, text: str) -> Tuple[int, int]:
        """Recommend optimal child and parent chunk sizes based on document analysis"""
        analysis = self.analyze_document(text)

        # Start with domain preset if detected with high confidence
        domain = analysis["domain"]
        if domain in self.presets:
            child_size = self.presets[domain]["child"]
            parent_size = self.presets[domain]["parent"]
            logger.info(f"✅ Using {domain} preset: child={child_size}, parent={parent_size}")
            return child_size, parent_size

        # Otherwise, calculate based on characteristics
        return self._calculate_sizes(analysis)

    def _detect_content_type(self, text: str) -> str:
        """Detect if academic, structured, or general"""
        academic_indicators = r"\[.*\]|\(.*et al\.|\\[a-z]+|^\d+\.|abstract|introduction|methodology"
        structured_indicators = r"^\s*[-*•]\s|^\d+\.|^#{1,6}\s|^#+\s"

        academic_score = len(re.findall(academic_indicators, text, re.IGNORECASE | re.MULTILINE))
        structured_score = len(re.findall(structured_indicators, text, re.MULTILINE))

        if academic_score > 10:
            return "academic"
        elif structured_score > 20:
            return "structured"
        return "general"

    def _detect_domain(self, text: str) -> Optional[str]:
        """Detect document domain (Wikipedia, academic, technical, etc.)"""
        text_lower = text.lower()

        # Wikipedia detection
        if "wikipedia" in text_lower or re.search(r"^(.*?)\n==.*==\n", text, re.MULTILINE):
            return "wikipedia"

        # Academic detection
        if re.search(r"abstract|introduction|methodology|conclusion|references", text_lower):
            return "academic"

        # Technical documentation
        if re.search(r"api|endpoint|parameter|return|function|class|method|usage", text_lower):
            return "technical_docs"

        # Code documentation
        if re.search(r'def |class |import |function|\\n\\s+"""', text):
            return "code_docs"

        # Blog/article
        if re.search(r"posted|author|updated|published|tagged", text_lower):
            return "blog"

        # News article
        if re.search(r"breaking|developing|source said|according to", text_lower):
            return "news"

        # Fiction (dialogue, narrative)
        if re.search(r'"[^"]*".*said|\'[^\']*\'.*thought', text):
            return "fiction"

        return None

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)"""
        # Factors: sentence length, vocabulary diversity, punctuation
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.5

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Score based on average sentence length
        if avg_sentence_length > 30:
            complexity_score = 0.9  # Very complex (academic)
        elif avg_sentence_length > 20:
            complexity_score = 0.7  # Complex
        elif avg_sentence_length > 12:
            complexity_score = 0.5  # Medium
        else:
            complexity_score = 0.3  # Simple

        # Adjust for special characters (equations, code)
        special_char_ratio = len(re.findall(r'[{}()[\]$\+=]', text)) / max(1, len(text))
        if special_char_ratio > 0.05:
            complexity_score = min(1.0, complexity_score + 0.2)

        return min(1.0, complexity_score)

    def _analyze_structure(self, text: str) -> float:
        """Analyze how well-structured the document is (0-1)"""
        # Look for section headers, lists, clear organization
        headers = len(re.findall(r'^#+\s', text, re.MULTILINE))
        lists = len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE))
        numbered = len(re.findall(r'^\d+\.\s', text, re.MULTILINE))

        structure_elements = headers + lists + numbered
        total_lines = len(text.split('\n'))

        if total_lines == 0:
            return 0.5

        structure_ratio = structure_elements / max(1, total_lines / 5)  # Normalize
        return min(1.0, structure_ratio)

    def _calculate_sizes(self, analysis: Dict) -> Tuple[int, int]:
        """Calculate chunk sizes based on analysis"""
        token_count = analysis["token_estimate"]
        complexity = analysis["complexity_score"]
        structure = analysis["structure_score"]
        content_type = analysis["content_type"]

        # Start with defaults
        base_child_size = 256
        base_parent_size = 1024

        # Adjust based on document length
        # Longer documents benefit from slightly larger chunks to maintain coherence
        if token_count > 10000:
            size_multiplier = 1.2
        elif token_count > 5000:
            size_multiplier = 1.1
        elif token_count < 1000:
            size_multiplier = 0.8
        else:
            size_multiplier = 1.0

        # Adjust based on complexity
        # Complex documents need larger chunks to maintain context
        if complexity > 0.8:
            complexity_multiplier = 1.3
        elif complexity > 0.6:
            complexity_multiplier = 1.1
        else:
            complexity_multiplier = 0.9

        # Adjust based on structure
        # Well-structured documents can use smaller chunks
        if structure > 0.7:
            structure_multiplier = 0.9
        else:
            structure_multiplier = 1.1

        # Apply multipliers
        child_size = int(base_child_size * size_multiplier * complexity_multiplier * structure_multiplier)
        parent_size = int(base_parent_size * size_multiplier * complexity_multiplier * structure_multiplier)

        # Ensure within reasonable bounds
        child_size = max(128, min(512, child_size))
        parent_size = max(512, min(2048, parent_size))

        # Ensure parent is 3-4x child size
        if parent_size < child_size * 3:
            parent_size = child_size * 3
        elif parent_size > child_size * 5:
            parent_size = child_size * 4

        return child_size, parent_size

    def get_detailed_recommendation(self, text: str) -> Dict:
        """Get detailed recommendations with explanations"""
        analysis = self.analyze_document(text)
        child_size, parent_size = self.recommend_chunk_sizes(text)

        recommendation = {
            "recommended_child_size": child_size,
            "recommended_parent_size": parent_size,
            "analysis": analysis,
            "reasoning": self._generate_reasoning(analysis, child_size, parent_size),
        }

        return recommendation

    @staticmethod
    def _generate_reasoning(analysis: Dict, child_size: int, parent_size: int) -> str:
        """Generate human-readable explanation for recommendations"""
        lines = []

        token_count = analysis["token_estimate"]
        complexity = analysis["complexity_score"]
        structure = analysis["structure_score"]
        domain = analysis["domain"]
        content_type = analysis["content_type"]

        lines.append(f"Document Analysis:")
        lines.append(f"  • Token estimate: {token_count:,} tokens")
        lines.append(f"  • Content type: {content_type}")
        if domain:
            lines.append(f"  • Detected domain: {domain}")
        lines.append(f"  • Complexity: {complexity:.0%} (0%=simple, 100%=complex)")
        lines.append(f"  • Structure: {structure:.0%} (0%=unstructured, 100%=well-organized)")

        lines.append(f"\nRecommended sizes:")
        lines.append(f"  • Child chunk: {child_size} tokens (precision retrieval)")
        lines.append(f"  • Parent chunk: {parent_size} tokens (context for LLM)")
        lines.append(f"  • Ratio: {parent_size/child_size:.1f}x")

        lines.append(f"\nWhy these sizes:")
        if complexity > 0.7:
            lines.append(f"  • Document is complex → larger chunks for better coherence")
        if structure < 0.5:
            lines.append(f"  • Document is unstructured → larger chunks for context")
        if token_count > 5000:
            lines.append(f"  • Document is long → slightly larger chunks to reduce fragmentation")
        elif token_count < 1000:
            lines.append(f"  • Document is short → smaller chunks for precision")

        return "\n".join(lines)

    def compare_presets(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Compare recommended sizes with all presets"""
        analysis = self.analyze_document(text)
        recommended = self.recommend_chunk_sizes(text)

        comparison = {
            "recommended": recommended,
            "presets": self.presets.copy(),
            "analysis": analysis,
        }

        return comparison
