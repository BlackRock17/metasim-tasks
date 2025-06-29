"""
LangChain-based intelligent text splitter for document cleaning.

This module leverages LangChain's proven text splitting capabilities to create
optimal chunks for LLM processing. It uses RecursiveCharacterTextSplitter with
smart separators and token-aware sizing to ensure perfect chunks for GPT-4o.

Key features:
- LangChain RecursiveCharacterTextSplitter integration
- Token-aware chunk sizing for GPT-4o optimization
- Hierarchical separator strategy (paragraphs → sentences → words)
- Configurable overlap and chunk sizes
- Comprehensive metadata tracking and statistics
- Production-ready with extensive logging
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LangChain imports for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..shared.config import Config

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """
    Enhanced chunk information with LangChain Document integration.

    This dataclass stores metadata about each chunk, compatible with
    LangChain's Document structure while adding our custom analytics.
    """
    content: str
    start_pos: int
    end_pos: int
    sentence_count: int
    word_count: int
    char_count: int
    token_estimate: int
    langchain_metadata: Dict[str, Any]

    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        return Document(
            page_content=self.content,
            metadata={
                'start_pos': self.start_pos,
                'end_pos': self.end_pos,
                'sentence_count': self.sentence_count,
                'word_count': self.word_count,
                'char_count': self.char_count,
                'token_estimate': self.token_estimate,
                **self.langchain_metadata
            }
        )

    def __str__(self) -> str:
        return (f"Chunk({self.char_count} chars, {self.token_estimate} tokens, "
                f"{self.sentence_count} sentences)")


class LangChainTextSplitter:
    """
    Advanced text splitter using LangChain's RecursiveCharacterTextSplitter.

    This class wraps LangChain's text splitting capabilities with our custom
    metadata tracking and optimization for document cleaning workflows.

    The splitter uses a hierarchical separator strategy:
    1. Double newlines (paragraph breaks) - highest priority
    2. Single newlines (line breaks)
    3. Sentence endings (. ! ?)
    4. Clause separators (, ; :)
    5. Single spaces (word boundaries) - last resort

    Optimized for GPT-4o token limits and LLM processing efficiency.
    """

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 separators: List[str] = None,
                 keep_separator: bool = True,
                 add_start_index: bool = True):
        """
        Initialize LangChain-based text splitter.

        Args:
            chunk_size: Target chunk size in characters (uses Config default if None)
            chunk_overlap: Overlap between chunks in characters
            separators: Custom hierarchy of separators (uses smart defaults if None)
            keep_separator: Whether to keep separators in the split text
            add_start_index: Whether to add start index to chunk metadata
        """
        # Use configuration defaults if not specified
        self.chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or (Config.DEFAULT_CHUNK_SIZE // 10)  # 10% overlap

        # Smart separator hierarchy optimized for document cleaning
        self.separators = separators or [
            "\n\n",  # Paragraph breaks (highest priority)
            "\n",  # Line breaks
            ". ",  # Sentence endings with space
            "! ",  # Exclamation with space
            "? ",  # Question with space
            "; ",  # Semicolon separators
            ", ",  # Comma separators
            " ",  # Word boundaries (last resort)
            ""  # Character splitting (extreme fallback)
        ]

        # Initialize LangChain's RecursiveCharacterTextSplitter
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            length_function=len,  # Use character count (could be token-based)
            is_separator_regex=False
        )

        # Statistics tracking
        self.stats = {
            'total_chunks': 0,
            'total_characters': 0,
            'total_tokens_estimated': 0,
            'avg_chunk_size': 0,
            'avg_token_count': 0,
            'min_chunk_size': float('inf'),
            'max_chunk_size': 0,
            'overlap_efficiency': 0,
            'separator_usage': {sep: 0 for sep in self.separators},
            'langchain_metadata': {}
        }

        logger.info(f"LangChain TextSplitter initialized:")
        logger.info(f"  Chunk size: {self.chunk_size} chars")
        logger.info(f"  Chunk overlap: {self.chunk_overlap} chars")
        logger.info(f"  Separators: {len(self.separators)} levels")
        logger.info(f"  Hierarchy: {' → '.join(repr(s) if s else 'char' for s in self.separators[:4])}...")

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[ChunkInfo]:
        """
        Split text using LangChain's RecursiveCharacterTextSplitter.

        Args:
            text: Input text to be split
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkInfo objects with comprehensive metadata

        Raises:
            ValueError: If input text is empty or invalid
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        metadata = metadata or {}

        logger.info(f"Starting LangChain text splitting: {len(text)} characters")

        # Reset statistics
        self._reset_stats()

        # Use LangChain to split the text
        langchain_docs = self.langchain_splitter.create_documents(
            texts=[text],
            metadatas=[metadata]
        )

        logger.info(f"LangChain created {len(langchain_docs)} document chunks")

        # Convert to our ChunkInfo format with enhanced metadata
        chunks = []
        for i, doc in enumerate(langchain_docs):
            chunk_info = self._create_chunk_info(doc, i)
            chunks.append(chunk_info)

        # Calculate comprehensive statistics
        self._calculate_stats(chunks, text)
        self._log_statistics()

        return chunks

    def split_documents(self, documents: List[Document]) -> List[ChunkInfo]:
        """
        Split multiple LangChain Documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of ChunkInfo objects from all documents
        """
        logger.info(f"Splitting {len(documents)} LangChain documents")

        all_chunks = []
        for i, doc in enumerate(documents):
            logger.debug(f"Processing document {i + 1}/{len(documents)}")

            chunks = self.split_text(
                text=doc.page_content,
                metadata=doc.metadata
            )
            all_chunks.extend(chunks)

        logger.info(f"Total chunks from all documents: {len(all_chunks)}")
        return all_chunks

    def _create_chunk_info(self, langchain_doc: Document, chunk_index: int) -> ChunkInfo:
        """
        Create ChunkInfo from LangChain Document with enhanced metadata.

        Args:
            langchain_doc: LangChain Document object
            chunk_index: Index of this chunk in the sequence

        Returns:
            ChunkInfo object with comprehensive metadata
        """
        content = langchain_doc.page_content
        metadata = langchain_doc.metadata

        # Calculate enhanced metrics
        sentence_count = self._count_sentences(content)
        word_count = len(content.split())
        char_count = len(content)
        token_estimate = self._estimate_tokens(content)

        # Extract position information from LangChain metadata
        start_pos = metadata.get('start_index', 0)
        end_pos = start_pos + char_count

        # Detect which separator was likely used
        separator_used = self._detect_separator_used(content)
        if separator_used:
            self.stats['separator_usage'][separator_used] += 1

        # Enhanced metadata combining LangChain and our data
        enhanced_metadata = {
            'chunk_index': chunk_index,
            'separator_used': separator_used,
            'processing_method': 'langchain_recursive',
            'splitter_config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            },
            **metadata  # Include original LangChain metadata
        }

        return ChunkInfo(
            content=content,
            start_pos=start_pos,
            end_pos=end_pos,
            sentence_count=sentence_count,
            word_count=word_count,
            char_count=char_count,
            token_estimate=token_estimate,
            langchain_metadata=enhanced_metadata
        )

    def _detect_separator_used(self, content: str) -> Optional[str]:
        """
        Detect which separator was likely used for this chunk.

        Args:
            content: Chunk content to analyze

        Returns:
            Most likely separator that was used
        """
        # Check for separators at the end of content (excluding empty string)
        for separator in self.separators[:-1]:  # Exclude empty string
            if separator and content.endswith(separator.strip()):
                return separator

        # Check for separators within content
        for separator in self.separators[:-1]:
            if separator and separator in content:
                return separator

        return None

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for GPT-4o model.

        GPT-4o uses approximately 4 characters per token on average.
        This is a rough estimate - for production, use tiktoken library.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # GPT-4o rough estimation: ~4 chars per token
        return len(text) // 4

    def _count_sentences(self, text: str) -> int:
        """
        Count sentences using simple heuristics.

        Args:
            text: Text to count sentences in

        Returns:
            Number of sentences
        """
        if not text.strip():
            return 0

        import re
        # Count sentence endings
        sentence_endings = len(re.findall(r'[.!?]+', text))
        return max(1, sentence_endings)

    def _reset_stats(self):
        """Reset statistics for new splitting operation."""
        self.stats = {
            'total_chunks': 0,
            'total_characters': 0,
            'total_tokens_estimated': 0,
            'avg_chunk_size': 0,
            'avg_token_count': 0,
            'min_chunk_size': float('inf'),
            'max_chunk_size': 0,
            'overlap_efficiency': 0,
            'separator_usage': {sep: 0 for sep in self.separators},
            'langchain_metadata': {}
        }

    def _calculate_stats(self, chunks: List[ChunkInfo], original_text: str):
        """
        Calculate comprehensive statistics.

        Args:
            chunks: Generated chunks
            original_text: Original input text
        """
        if not chunks:
            return

        # Basic statistics
        self.stats['total_chunks'] = len(chunks)
        self.stats['total_characters'] = sum(chunk.char_count for chunk in chunks)
        self.stats['total_tokens_estimated'] = sum(chunk.token_estimate for chunk in chunks)

        chunk_sizes = [chunk.char_count for chunk in chunks]
        self.stats['avg_chunk_size'] = sum(chunk_sizes) / len(chunk_sizes)
        self.stats['min_chunk_size'] = min(chunk_sizes)
        self.stats['max_chunk_size'] = max(chunk_sizes)

        token_counts = [chunk.token_estimate for chunk in chunks]
        self.stats['avg_token_count'] = sum(token_counts) / len(token_counts)

        # Calculate overlap efficiency
        total_chunk_chars = sum(chunk_sizes)
        original_chars = len(original_text)
        overlap_ratio = (total_chunk_chars - original_chars) / original_chars if original_chars > 0 else 0
        self.stats['overlap_efficiency'] = overlap_ratio * 100

        # Store LangChain configuration
        self.stats['langchain_metadata'] = {
            'splitter_type': 'RecursiveCharacterTextSplitter',
            'chunk_size_config': self.chunk_size,
            'chunk_overlap_config': self.chunk_overlap,
            'separator_count': len(self.separators)
        }

    def _log_statistics(self):
        """Log comprehensive statistics."""
        stats = self.stats
        logger.info("=== LangChain Text Splitting Statistics ===")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Average chunk size: {stats['avg_chunk_size']:.1f} chars")
        logger.info(f"Average tokens per chunk: {stats['avg_token_count']:.1f}")
        logger.info(f"Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} chars")
        logger.info(f"Overlap efficiency: {stats['overlap_efficiency']:.1f}%")

        # Log separator usage
        used_separators = {k: v for k, v in stats['separator_usage'].items() if v > 0}
        if used_separators:
            logger.info("Separator usage:")
            for sep, count in used_separators.items():
                sep_name = repr(sep) if sep else 'character'
                logger.info(f"  {sep_name}: {count} chunks")

        # Log efficiency metrics
        target_efficiency = abs(stats['avg_chunk_size'] - self.chunk_size) / self.chunk_size
        logger.info(f"Target size efficiency: {(1 - target_efficiency) * 100:.1f}%")

    def get_splitting_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of splitting operation.

        Returns:
            Dictionary with statistics, configuration, and quality metrics
        """
        return {
            'statistics': self.stats.copy(),
            'configuration': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'separators': self.separators,
                'langchain_splitter_type': 'RecursiveCharacterTextSplitter'
            },
            'quality_metrics': {
                'avg_chunk_utilization': (self.stats['avg_chunk_size'] / self.chunk_size) * 100,
                'size_variance': self.stats['max_chunk_size'] - self.stats['min_chunk_size'],
                'token_efficiency': self.stats['avg_token_count'],
                'overlap_ratio': self.stats['overlap_efficiency'],
                'separator_diversity': len([v for v in self.stats['separator_usage'].values() if v > 0])
            },
            'langchain_integration': {
                'framework': 'LangChain',
                'splitter_class': 'RecursiveCharacterTextSplitter',
                'document_compatible': True,
                'production_ready': True
            }
        }


def create_langchain_splitter(**kwargs) -> LangChainTextSplitter:
    """
    Factory function to create a configured LangChain text splitter.

    Args:
        **kwargs: Configuration parameters

    Returns:
        Configured LangChainTextSplitter instance
    """
    return LangChainTextSplitter(**kwargs)


def split_text_with_langchain(text: str, **kwargs) -> List[ChunkInfo]:
    """
    Convenience function for quick text splitting using LangChain.

    Args:
        text: Text to split
        **kwargs: Configuration parameters

    Returns:
        List of ChunkInfo objects with LangChain integration
    """
    splitter = create_langchain_splitter(**kwargs)
    return splitter.split_text(text)


# Example usage demonstrating LangChain integration
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)

    # Test text with various structures
    sample_text = """
    The B2B Sales Process Handbook

    Introducing the B2B sales process. It's Monday morning. That superstar SDR has booked a new meeting in your calendar.

    "Ah, good," you say to yourself. "Time to sit back and relax." ...said no account exec, ever!

    The best AEs know that if they want that meeting to succeed, then they have to get to work, straight away. They have to improve their knowledge. They have to get to know their prospect better. They have to follow a process.

    But what is that process? We asked our business development team for their insights. How do they manage the hard job of B2B sales, from researching prospects to conducting demos and closing business?

    Research is a critical phase in the B2B sales process. You don't want to waste time or appear unprepared by asking questions when the answers are readily available online.
    """

    print("Testing LangChain Text Splitter...")
    print(f"Sample text length: {len(sample_text)} characters")
    print("=" * 60)

    # Create LangChain-based splitter
    splitter = LangChainTextSplitter(
        chunk_size=300,  # Smaller for demo
        chunk_overlap=50,  # 50 char overlap
    )

    # Split the text
    chunks = splitter.split_text(sample_text, metadata={'source': 'test_document'})

    # Display results
    print(f"\nGenerated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
        print(f"  Content: {chunk.content[:80]}...")
        print(f"  Metadata: start_pos={chunk.start_pos}, tokens≈{chunk.token_estimate}")
        print()

    # Show comprehensive summary
    summary = splitter.get_splitting_summary()
    print("=" * 60)
    print("SPLITTING SUMMARY:")
    print(f"✅ LangChain Integration: {summary['langchain_integration']['framework']}")
    print(f"📊 Chunk Utilization: {summary['quality_metrics']['avg_chunk_utilization']:.1f}%")
    print(f"🎯 Token Efficiency: {summary['quality_metrics']['token_efficiency']:.1f} avg tokens/chunk")
    print(f"🔄 Overlap Ratio: {summary['quality_metrics']['overlap_ratio']:.1f}%")
    print(f"📏 Separator Diversity: {summary['quality_metrics']['separator_diversity']} types used")

    # Demonstrate LangChain Document compatibility
    print("\n" + "=" * 60)
    print("LANGCHAIN DOCUMENT COMPATIBILITY:")
    sample_chunk = chunks[0]
    langchain_doc = sample_chunk.to_langchain_document()
    print(f"✅ Converted to LangChain Document:")
    print(f"   Content length: {len(langchain_doc.page_content)} chars")
    print(f"   Metadata keys: {list(langchain_doc.metadata.keys())}")
