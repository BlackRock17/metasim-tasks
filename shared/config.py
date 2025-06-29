"""
Simple configuration settings for the document cleaner application.

Contains only essential settings needed for the interview tasks.
All complex features and utilities have been removed for clarity.
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
    DEFAULT_CHUNK_SIZE = 1500
    MAX_CHUNK_SIZE = 2000
    MIN_CHUNK_SIZE = 200

    # Network Configuration
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3


# Convenience function for getting config
def get_config() -> Config:
    """Get the configuration class."""
    return Config
