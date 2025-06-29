"""
Simple but professional text splitter using LangChain.

This module provides text chunking functionality that preserves sentence
boundaries using LangChain's RecursiveCharacterTextSplitter.

Key features:
- LangChain integration (as required by the task)
- Sentence boundary preservation
- Configurable chunk sizes
- Basic error handling and logging
"""

import logging
from typing import List
from dataclasses import dataclass

# LangChain import for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import Config

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """
    Simple chunk information container.

    Contains just the essential information needed for document processing.
    """
    content: str
    start_pos: int
    char_count: int

    def __str__(self) -> str:
        return f"Chunk({self.char_count} chars)"


class TextSplitter:
    """
    Simple text splitter using LangChain's RecursiveCharacterTextSplitter.

    This class provides an easy-to-use interface for splitting text into
    chunks while preserving sentence boundaries.
    """

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        # Use config defaults if not specified
        self.chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or (self.chunk_size // 10)  # 10% overlap

        # Smart separator hierarchy for better splitting
        self.separators = [
            "\n\n",  # Paragraph breaks (highest priority)
            "\n",  # Line breaks
            ". ",  # Sentence endings
            "! ",  # Exclamation sentences
            "? ",  # Question sentences
            "; ",  # Semicolon breaks
            ", ",  # Comma breaks
            " ",  # Word boundaries (last resort)
            ""  # Character splitting (extreme fallback)
        ]

        # Initialize LangChain splitter
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
            add_start_index=True,
            length_function=len
        )

        logger.info(f"TextSplitter initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def split_text(self, text: str) -> List[ChunkInfo]:
        """
        Split text into chunks using LangChain.

        Args:
            text: Input text to be split

        Returns:
            List of ChunkInfo objects containing the split text

        Raises:
            ValueError: If input text is empty
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        logger.info(f"Splitting text: {len(text)} characters")

        # Use LangChain to split the text
        langchain_docs = self.langchain_splitter.create_documents([text])

        # Convert to our ChunkInfo format
        chunks = []
        for i, doc in enumerate(langchain_docs):
            chunk_info = ChunkInfo(
                content=doc.page_content,
                start_pos=doc.metadata.get('start_index', 0),
                char_count=len(doc.page_content)
            )
            chunks.append(chunk_info)

        logger.info(f"Created {len(chunks)} chunks")

        # Log basic statistics
        if chunks:
            avg_size = sum(chunk.char_count for chunk in chunks) / len(chunks)
            min_size = min(chunk.char_count for chunk in chunks)
            max_size = max(chunk.char_count for chunk in chunks)
            logger.info(f"Chunk sizes: avg={avg_size:.0f}, range={min_size}-{max_size}")

        return chunks


def create_text_splitter(**kwargs) -> TextSplitter:
    """
    Factory function to create a configured text splitter.

    Args:
        **kwargs: Configuration parameters

    Returns:
        Configured TextSplitter instance
    """
    return TextSplitter(**kwargs)


def split_text(text: str, **kwargs) -> List[ChunkInfo]:
    """
    Convenience function for quick text splitting.

    Args:
        text: Text to split
        **kwargs: Configuration parameters

    Returns:
        List of text chunks
    """
    splitter = create_text_splitter(**kwargs)
    return splitter.split_text(text)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)

    # Test with sample text
    sample_text = """
    This is a test document with multiple paragraphs.
    It should be split intelligently by the LangChain splitter.

    This is a second paragraph with more content to test
    the splitting functionality. The algorithm should handle
    various text structures gracefully.

    Short paragraph.

    This is a longer paragraph that might need to be split
    if it exceeds the target chunk size. It contains multiple
    sentences to test sentence boundary preservation.
    """

    print("Testing TextSplitter...")
    print(f"Sample text length: {len(sample_text)} characters")
    print("-" * 50)

    # Create splitter with small chunk size for demo
    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)

    # Split the text
    chunks = splitter.split_text(sample_text)

    # Display results
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}: {chunk}")
        print(f"Content: {chunk.content[:80]}...")

    print(f"\nTotal chunks created: {len(chunks)}")
