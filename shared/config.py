"""
Configuration settings for the document cleaner application.

This module centralizes all configuration parameters to make the application
easily configurable and maintainable. All settings can be modified here
without changing the core logic.
"""

import os
from pathlib import Path


class Config:
    """
    Central configuration class containing all application settings.

    This class uses class variables to store configuration parameters,
    making them easily accessible throughout the application without
    needing to instantiate objects.
    """


    # FastAPI Server Configuration

    # Base URL for the FastAPI server (Task 1 implementation)
    FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

    # Specific endpoint for text cleaning functionality
    CLEAN_TEXT_ENDPOINT = "/clean-text"

    # Health check endpoint to verify server is running
    HEALTH_ENDPOINT = "/health"


    # Text Processing Configuration

    # Optimal chunk size for text processing (in characters)
    # This size balances between:
    # - Having enough context for the LLM to understand the text
    # - Not exceeding the LLM's token limits
    # - Maintaining processing efficiency
    DEFAULT_CHUNK_SIZE = 1500

    # Maximum allowed chunk size before forcing a split
    MAX_CHUNK_SIZE = 2000

    # Minimum chunk size to maintain context
    MIN_CHUNK_SIZE = 200

    # Characters used for intelligent text splitting (in order of preference)
    SPLIT_CHARACTERS = [
        "\n\n",  # Paragraph breaks (highest priority)
        ". ",  # Sentence endings
        "! ",  # Exclamation sentence endings
        "? ",  # Question sentence endings
        "; ",  # Semicolon breaks
        ", ",  # Comma breaks
        " ",  # Word breaks (last resort)
    ]


    # Network and API Configuration

    # Timeout for HTTP requests to FastAPI server (in seconds)
    REQUEST_TIMEOUT = 30

    # Number of retry attempts for failed API calls
    MAX_RETRIES = 3

    # Delay between retry attempts (in seconds)
    RETRY_DELAY = 1


    # File System Configuration

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent

    # Directory containing sample documents for testing
    SAMPLE_DOCUMENTS_DIR = PROJECT_ROOT / "document_cleaner" / "sample_documents"

    # Default input file (the noisy B2B text)
    DEFAULT_INPUT_FILE = SAMPLE_DOCUMENTS_DIR / "b2b_extracted_text_with_noise.txt"

    # Reference clean file for comparison
    REFERENCE_CLEAN_FILE = SAMPLE_DOCUMENTS_DIR / "b2b_extracted_text.txt"

    # Default output directory
    OUTPUT_DIR = PROJECT_ROOT / "output"

    # Default output filename
    DEFAULT_OUTPUT_FILE = "cleaned_document.txt"


    # Logging Configuration

    # Logging level for the application
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Enable/disable verbose logging
    VERBOSE_LOGGING = os.getenv("VERBOSE", "false").lower() == "true"


    # Processing Configuration

    # Enable/disable parallel processing of chunks
    ENABLE_PARALLEL_PROCESSING = False  # Start with sequential for simplicity

    # Maximum number of concurrent API requests (if parallel processing enabled)
    MAX_CONCURRENT_REQUESTS = 3

    # Enable/disable progress bar display
    SHOW_PROGRESS_BAR = True


    # Quality Control Configuration

    # Enable automatic comparison with reference file (if available)
    ENABLE_QUALITY_CHECK = True

    # Minimum expected cleaning improvement (percentage)
    # Used to validate that the cleaning process is working
    MIN_CLEANING_IMPROVEMENT = 0.05  # 5% reduction in artifacts


    # Utility Methods

    @classmethod
    def get_full_api_url(cls, endpoint: str = None) -> str:
        """
        Construct the full API URL for a given endpoint.

        Args:
            endpoint (str, optional): API endpoint. Defaults to clean-text endpoint.

        Returns:
            str: Complete API URL

        Example:
            >>> Config.get_full_api_url()
            'http://localhost:8000/clean-text'
            >>> Config.get_full_api_url('/health')
            'http://localhost:8000/health'
        """
        if endpoint is None:
            endpoint = cls.CLEAN_TEXT_ENDPOINT

        # Ensure endpoint starts with '/'
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint

        return f"{cls.FASTAPI_BASE_URL}{endpoint}"

    @classmethod
    def ensure_output_dir(cls) -> Path:
        """
        Ensure the output directory exists, create it if necessary.

        Returns:
            Path: Path to the output directory
        """
        output_path = cls.OUTPUT_DIR
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @classmethod
    def get_output_filepath(cls, filename: str = None) -> Path:
        """
        Get the full path for an output file.

        Args:
            filename (str, optional): Output filename. Defaults to default output file.

        Returns:
            Path: Complete path to the output file
        """
        if filename is None:
            filename = cls.DEFAULT_OUTPUT_FILE

        return cls.ensure_output_dir() / filename

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate the current configuration settings.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check chunk size constraints
        if not (cls.MIN_CHUNK_SIZE <= cls.DEFAULT_CHUNK_SIZE <= cls.MAX_CHUNK_SIZE):
            print(f"ERROR: Invalid chunk size configuration. "
                  f"Min: {cls.MIN_CHUNK_SIZE}, Default: {cls.DEFAULT_CHUNK_SIZE}, "
                  f"Max: {cls.MAX_CHUNK_SIZE}")
            return False

        # Check if sample documents directory exists
        if not cls.SAMPLE_DOCUMENTS_DIR.exists():
            print(f"WARNING: Sample documents directory not found: {cls.SAMPLE_DOCUMENTS_DIR}")

        # Check if default input file exists
        if not cls.DEFAULT_INPUT_FILE.exists():
            print(f"WARNING: Default input file not found: {cls.DEFAULT_INPUT_FILE}")

        return True


# Module-level convenience functions

def get_config() -> Config:
    """
    Get the configuration instance.

    Returns:
        Config: The configuration class
    """
    return Config


def print_config_summary():
    """
    Print a summary of the current configuration.
    Useful for debugging and verification.
    """
    print("=== Document Cleaner Configuration ===")
    print(f"FastAPI URL: {Config.FASTAPI_BASE_URL}")
    print(f"Clean Text Endpoint: {Config.get_full_api_url()}")
    print(f"Chunk Size: {Config.DEFAULT_CHUNK_SIZE} chars (min: {Config.MIN_CHUNK_SIZE}, max: {Config.MAX_CHUNK_SIZE})")
    print(f"Request Timeout: {Config.REQUEST_TIMEOUT}s")
    print(f"Max Retries: {Config.MAX_RETRIES}")
    print(f"Input File: {Config.DEFAULT_INPUT_FILE}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")
    print(f"Log Level: {Config.LOG_LEVEL}")
    print("=" * 40)


# Validate configuration when module is imported
if __name__ == "__main__":
    # This block runs only when the file is executed directly
    print_config_summary()

    if Config.validate_config():
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has issues!")
