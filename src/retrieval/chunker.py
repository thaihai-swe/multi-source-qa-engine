"""Adaptive chunking strategy for various content types"""
from abc import ABC
from typing import List, Tuple, Dict, Any
import re
from src.retrieval import Chunker
from src.models import ContentType
from src.retrieval.smart_chunker import SmartChunkSizer
from src.utils import get_logger

logger = get_logger()


class ChunkHierarchy:
    """Tracks parent-child relationships between chunks"""
    def __init__(self):
        self.parent_chunks: Dict[str, str] = {}  # parent_id -> full content
        self.child_parents: Dict[str, str] = {}  # child_id -> parent_id
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata

    def add_parent_child(self, parent_id: str, parent_content: str,
                        child_ids: List[str], child_contents: List[str]) -> None:
        """Register parent-child relationships"""
        self.parent_chunks[parent_id] = parent_content
        for child_id in child_ids:
            self.child_parents[child_id] = parent_id
            self.chunk_metadata[child_id] = {"parent_id": parent_id}

    def get_parent_content(self, child_id: str) -> str:
        """Get parent content for a child chunk"""
        parent_id = self.child_parents.get(child_id)
        if parent_id:
            return self.parent_chunks.get(parent_id, "")
        return ""

    def get_hierarchy_info(self, chunk_id: str) -> Dict[str, Any]:
        """Get hierarchy info for a chunk"""
        if chunk_id in self.child_parents:
            return {"type": "child", "parent_id": self.child_parents[chunk_id]}
        elif chunk_id in self.parent_chunks:
            return {"type": "parent"}
        return {"type": "independent"}


class AdaptiveChunker(Chunker):
    """Content-aware chunking strategy that adapts based on content type"""

    def __init__(
        self,
        default_chunk_size: int = 512,
        default_overlap: int = 50,
        enable_hierarchy: bool = False,
        # Parent-child configuration
        child_chunk_size: int = 256,
        parent_chunk_size: int = 1024,
        enable_smart_sizing: bool = False,
    ):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.enable_hierarchy = enable_hierarchy
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.enable_smart_sizing = enable_smart_sizing
        self.hierarchy = ChunkHierarchy()
        self.smart_sizer = SmartChunkSizer() if enable_smart_sizing else None
        self.last_sizing_info = None

    def chunk(self, text: str) -> List[str]:
        """Adaptively chunk based on content type"""
        if self.enable_hierarchy:
            # Return only child chunks; parents are tracked in self.hierarchy
            return self.chunk_with_hierarchy(text)
        else:
            content_type = self._detect_content_type(text)
            chunk_size, overlap = self._get_optimal_params(content_type)
            return self._chunk_with_overlap(text, chunk_size, overlap)

    def chunk_with_hierarchy(self, text: str) -> List[str]:
        """Create child chunks with parent reference tracking"""
        # Determine chunk sizes (smart or fixed)
        if self.enable_smart_sizing and self.smart_sizer:
            child_size, parent_size = self.smart_sizer.recommend_chunk_sizes(text)
            sizing_info = self.smart_sizer.get_detailed_recommendation(text)
            self.last_sizing_info = sizing_info
            logger.info(f"📊 Smart chunking analysis: {sizing_info['analysis']['content_type']} document")
            logger.info(f"🎯 Recommended sizes: child={child_size}, parent={parent_size}")
        else:
            child_size = self.child_chunk_size
            parent_size = self.parent_chunk_size

        # First, create small child chunks for precision retrieval
        child_chunks = self._chunk_with_overlap(text, child_size,
                                                overlap=int(child_size * 0.1))

        # Then, create larger parent chunks for context
        parent_chunks = self._chunk_with_overlap(text, parent_size,
                                                 overlap=int(parent_size * 0.15))

        # Build hierarchy: map each child to its closest parent
        self._build_hierarchy(child_chunks, parent_chunks, text)

        return child_chunks

    def _build_hierarchy(self, child_chunks: List[str], parent_chunks: List[str],
                        full_text: str) -> None:
        """Build parent-child relationships"""
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            parent_id = f"parent_{parent_idx}"
            child_ids_for_parent = []

            # Find child chunks that fall within this parent chunk's span
            parent_start = full_text.find(parent_chunk)
            if parent_start == -1:
                parent_start = parent_idx * self.parent_chunk_size

            parent_end = parent_start + len(parent_chunk)

            for child_idx, child_chunk in enumerate(child_chunks):
                child_start = full_text.find(child_chunk)
                if child_start == -1:
                    child_start = child_idx * self.child_chunk_size

                # Check if child overlaps with parent's range
                if parent_start <= child_start < parent_end:
                    child_ids_for_parent.append(f"parent_{parent_idx}_child_{child_idx}")

            if child_ids_for_parent:
                self.hierarchy.add_parent_child(
                    parent_id, parent_chunk, child_ids_for_parent, child_chunks
                )

    def get_parent_for_chunk(self, chunk_id: str) -> str:
        """Retrieve parent chunk content for a given child chunk ID"""
        return self.hierarchy.get_parent_content(chunk_id)

    def get_hierarchy(self) -> ChunkHierarchy:
        """Get the chunk hierarchy tracker"""
        return self.hierarchy

    def get_sizing_info(self) -> Dict:
        """Get last smart sizing analysis results"""
        return self.last_sizing_info or {}

    @staticmethod
    def _detect_content_type(text: str) -> str:
        """Detect content type based on text characteristics"""
        # Academic: citations, equations, formal structure
        if re.search(r"\[.*\]|\(.*et al\.|\\[a-z]+|^\d+\.", text, re.IGNORECASE):
            return "academic"
        # Structured: tables, lists, clear hierarchy
        if re.search(r"^\s*[-*•]\s|^\d+\.|^#{1,6}\s", text, re.MULTILINE):
            return "structured"
        return "general"

    def _get_optimal_params(self, content_type: str) -> tuple[int, int]:
        """Get optimal chunk size and overlap for content type"""
        if content_type == "academic":
            return (1024, 200)  # Larger chunks, more overlap
        elif content_type == "structured":
            return (256, 32)  # Smaller chunks, less overlap
        else:
            return (512, 50)  # Default

    @staticmethod
    def _chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
