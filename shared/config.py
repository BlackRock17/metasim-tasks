"""
Simple configuration settings for the document cleaner application.

Contains only essential settings needed for the tasks.
"""

import os


class Config:
    """
    Simple configuration class with only essential settings.

    This class contains the minimum configuration needed to complete
    the interview tasks successfully.
    """

    # FastAPI Server Configuration
    FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

    # Text Processing Configuration
    # Chunk size optimized based on GPT-4o cleaning prompt performance testing:
    # - 500 chars: Insufficient context for effective artifact removal
    # - 1000 chars: Good results but loses some semantic coherence
    # - 1500 chars: Optimal balance - maintains context while fitting prompt limits
    # - 2000+ chars: Approach token limits, risk of incomplete processing
    DEFAULT_CHUNK_SIZE = 1500
    MAX_CHUNK_SIZE = 2000
    MIN_CHUNK_SIZE = 200

    # Network Configuration
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

    # File Paths Configuration
    DATA_INPUT_DIR = "data/input"
    DATA_OUTPUT_DIR = "data/output"
    DATA_SAMPLES_DIR = "data/samples"


# Convenience function for getting config
def get_config() -> Config:
    """Get the configuration class."""
    return Config
