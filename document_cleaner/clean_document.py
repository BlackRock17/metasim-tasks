"""
Simple but professional document cleaner for removing text artifacts.

This script implements the core requirements from Task 2:
1. Load input document from file
2. Split into chunks using LangChain text splitter
3. Send each chunk to FastAPI cleaning service
4. Collect and aggregate cleaned results
5. Save final cleaned document to output file

Features:
- LangChain text splitting (as required)
- Basic error handling with retry logic
- Simple progress tracking
- Clean, readable code structure
"""

import os
import time
import logging
from pathlib import Path
from typing import List

# Import our custom modules
from .text_splitter import TextSplitter, ChunkInfo
from ..shared.api_client import FastAPIClient, APIClientError
from ..shared.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DocumentCleaningError(Exception):
    """Custom exception for document cleaning errors."""
    pass


class DocumentCleaner:
    """
    Simple document cleaner that coordinates the cleaning workflow.

    This class handles the complete process from loading text to producing
    clean output, with basic error handling and progress tracking.
    """

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 max_retries: int = 3):
        """
        Initialize the document cleaner.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            max_retries: Maximum retry attempts for API calls
        """
        # Initialize components
        self.api_client = FastAPIClient()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.max_retries = max_retries

        logger.info("DocumentCleaner initialized")

    def clean_document(self, input_file: str, output_file: str = None) -> dict:
        """
        Clean a document by removing artifacts and noise.

        Args:
            input_file: Path to input text file
            output_file: Path to output file (auto-generated if None)

        Returns:
            Dictionary with cleaning results

        Raises:
            DocumentCleaningError: If cleaning process fails
        """
        start_time = time.time()

        try:
            logger.info(f"Starting document cleaning: {input_file}")

            # Step 1: Load document
            original_text = self._load_document(input_file)
            logger.info(f"Loaded document: {len(original_text)} characters")

            # Step 2: Split into chunks
            chunks = self._split_text(original_text)
            logger.info(f"Split into {len(chunks)} chunks")

            # Step 3: Clean chunks via API
            cleaned_chunks = self._clean_chunks(chunks)

            # Step 4: Aggregate results
            cleaned_text = self._aggregate_chunks(cleaned_chunks)

            # Step 5: Save cleaned document
            if output_file is None:
                output_file = self._generate_output_filename(input_file)

            self._save_document(cleaned_text, output_file)

            # Calculate results
            processing_time = time.time() - start_time

            logger.info("=" * 50)
            logger.info("CLEANING COMPLETED SUCCESSFULLY")
            logger.info(f"Input:  {input_file}")
            logger.info(f"Output: {output_file}")
            logger.info(f"Original: {len(original_text)} characters")
            logger.info(f"Cleaned:  {len(cleaned_text)} characters")
            logger.info(f"Time: {processing_time:.1f} seconds")
            logger.info("=" * 50)

            return {
                'success': True,
                'input_file': input_file,
                'output_file': output_file,
                'original_length': len(original_text),
                'cleaned_length': len(cleaned_text),
                'processing_time': processing_time,
                'chunks_processed': len(chunks)
            }

        except Exception as e:
            logger.error(f"Document cleaning failed: {e}")
            raise DocumentCleaningError(f"Failed to clean document: {e}") from e

    def _load_document(self, input_file: str) -> str:
        """
        Load document from file with basic error handling.

        Args:
            input_file: Path to input file

        Returns:
            Text content of the file
        """
        input_path = Path(input_file)

        if not input_path.exists():
            raise DocumentCleaningError(f"Input file not found: {input_file}")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                raise DocumentCleaningError("Input file is empty")

            return content

        except UnicodeDecodeError as e:
            raise DocumentCleaningError(f"Cannot read file (encoding error): {e}")
        except IOError as e:
            raise DocumentCleaningError(f"Cannot read file: {e}")

    def _split_text(self, text: str) -> List[ChunkInfo]:
        """
        Split text into chunks using the text splitter.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            raise DocumentCleaningError(f"Text splitting failed: {e}")

    def _clean_chunks(self, chunks: List[ChunkInfo]) -> List[str]:
        """
        Clean each chunk by sending it to the FastAPI service.

        Args:
            chunks: List of chunks to clean

        Returns:
            List of cleaned text strings
        """
        cleaned_chunks = []
        successful = 0
        failed = 0

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)} ({chunk.char_count} chars)")

            try:
                # Clean this chunk with retry
                cleaned_text = self._clean_single_chunk(chunk)
                cleaned_chunks.append(cleaned_text)
                successful += 1

            except APIClientError as e:
                logger.warning(f"Failed to clean chunk {i}: {e}")
                logger.warning("Using original text as fallback")

                # Use original text as fallback
                cleaned_chunks.append(chunk.content)
                failed += 1

        logger.info(f"Cleaning completed: {successful} successful, {failed} failed")

        # Check if too many failures
        if failed > len(chunks) // 2:  # More than half failed
            raise DocumentCleaningError(
                f"Too many cleaning failures ({failed}/{len(chunks)}). "
                f"Check if FastAPI server is running."
            )

        return cleaned_chunks

    def _clean_single_chunk(self, chunk: ChunkInfo) -> str:
        """
        Clean a single chunk with retry logic.

        Args:
            chunk: Chunk to clean

        Returns:
            Cleaned text
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Simple delay for retry
                    delay = attempt * 1.0
                    logger.debug(f"Retrying after {delay}s...")
                    time.sleep(delay)

                # Make API call
                cleaned_text = self.api_client.clean_text(chunk.content)

                if not cleaned_text.strip():
                    raise APIClientError("API returned empty text")

                return cleaned_text

            except APIClientError as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.debug(f"Attempt {attempt + 1} failed, retrying...")

        # All attempts failed
        raise last_error

    def _aggregate_chunks(self, cleaned_chunks: List[str]) -> str:
        """
        Combine cleaned chunks into final document.

        Args:
            cleaned_chunks: List of cleaned text pieces

        Returns:
            Final aggregated text
        """
        # Simple aggregation: join with space
        aggregated = " ".join(chunk.strip() for chunk in cleaned_chunks if chunk.strip())

        # Basic cleanup
        import re
        aggregated = re.sub(r' +', ' ', aggregated)  # Multiple spaces -> single
        aggregated = re.sub(r'\.([A-Z])', r'. \1', aggregated)  # Space after period

        return aggregated.strip()

    def _save_document(self, text: str, output_file: str):
        """
        Save cleaned text to output file.

        Args:
            text: Cleaned text to save
            output_file: Output file path
        """
        try:
            output_path = Path(output_file)

            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            logger.info(f"Saved cleaned document: {output_path}")

        except IOError as e:
            raise DocumentCleaningError(f"Cannot save output file: {e}")

    def _generate_output_filename(self, input_file: str) -> str:
        """
        Generate output filename from input filename.

        Args:
            input_file: Input file path

        Returns:
            Generated output file path
        """
        input_path = Path(input_file)
        stem = input_path.stem
        suffix = input_path.suffix or '.txt'

        # Generate: original_name_cleaned.txt
        output_filename = f"{stem}_cleaned{suffix}"

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        return str(output_path)


def clean_document_file(input_file: str, output_file: str = None, **kwargs) -> dict:
    """
    Convenience function to clean a document file.

    Args:
        input_file: Path to input file
        output_file: Path to output file (auto-generated if None)
        **kwargs: Additional configuration

    Returns:
        Cleaning results dictionary
    """
    cleaner = DocumentCleaner(**kwargs)
    return cleaner.clean_document(input_file, output_file)


def main():
    """
    Simple command-line interface for document cleaning.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python clean_document.py input_file.txt [output_file.txt]")
        print("Example: python clean_document.py document.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    try:
        print(f"Cleaning document: {input_file}")
        print("This may take a few minutes...")
        print()

        # Create cleaner and process
        cleaner = DocumentCleaner()
        result = cleaner.clean_document(input_file, output_file)

        print()
        print("SUCCESS! Document cleaned successfully.")
        print(f"Check the output file: {result['output_file']}")

    except DocumentCleaningError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCleaning interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
