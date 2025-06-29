"""
Simple HTTP client for communicating with the FastAPI server.

This module provides basic functionality for making API calls to the FastAPI server
for text cleaning. Simplified for interview clarity.
"""

import requests
import time
import logging
from typing import Optional, Dict, Any

from .config import Config

# Setup logging
logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Simple exception for API client errors."""
    pass


class FastAPIClient:
    """
    Simple HTTP client for the FastAPI text cleaning service.

    Provides basic text cleaning functionality with simple error handling and retries.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        """Initialize the FastAPI client."""
        self.base_url = base_url or Config.FASTAPI_BASE_URL
        self.timeout = timeout or Config.REQUEST_TIMEOUT
        logger.info(f"FastAPI client initialized with base URL: {self.base_url}")

    def health_check(self) -> bool:
        """Check if the FastAPI server is running."""
        try:
            health_url = f"{Config.FASTAPI_BASE_URL}/health"
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            logger.info("Server health check passed")
            return True
        except Exception as e:
            logger.warning(f"Server health check failed: {e}")
            return False

    def clean_text(self, text: str) -> str:
        """
        Send text to the FastAPI server for cleaning.

        Args:
            text (str): Text to be cleaned

        Returns:
            str: Cleaned text from the API

        Raises:
            APIClientError: If the API call fails after all retries
        """
        if not text or not text.strip():
            return text

        payload = {"text": text}

        # Simple retry logic
        for attempt in range(Config.MAX_RETRIES):
            try:
                return self._make_request(payload, attempt)
            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)  # Simple 1 second delay
                else:
                    logger.error(f"All attempts failed: {e}")
                    raise APIClientError(f"Text cleaning failed: {e}")

    def _make_request(self, payload: Dict[str, Any], attempt: int) -> str:
        """Make a single HTTP request to the clean-text endpoint."""
        clean_text_url = f"{Config.FASTAPI_BASE_URL}/clean-text"

        response = requests.post(
            clean_text_url,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        response_data = response.json()

        if "cleaned_text" not in response_data:
            raise APIClientError("Missing 'cleaned_text' in API response")

        return response_data["cleaned_text"]


def create_api_client() -> FastAPIClient:
    """Factory function to create a configured FastAPI client."""
    return FastAPIClient()


def test_api_connection() -> bool:
    """Test the connection to the FastAPI server."""
    print("Testing FastAPI server connection...")

    try:
        client = create_api_client()

        # Test health check
        if not client.health_check():
            print("❌ Health check failed")
            return False

        # Test text cleaning
        test_text = "Hello world\\n\\nPage 1\\n\\nThis is a test."
        try:
            cleaned = client.clean_text(test_text)
            print(f"✅ Text cleaning test successful")
            print(f"   Original: {repr(test_text)}")
            print(f"   Cleaned:  {repr(cleaned)}")
            return True
        except APIClientError as e:
            print(f"❌ Text cleaning test failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)

    # Run connection test
    success = test_api_connection()

    if success:
        print("\n🎉 All tests passed! API client is ready to use.")
    else:
        print("\n💥 Tests failed. Please check:")
        print("   1. FastAPI server is running (python -m fastapi_server.main)")
        print(f"   2. Server URL is correct: {Config.FASTAPI_BASE_URL}")
        print("   3. Network connectivity is working")
