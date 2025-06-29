"""
HTTP client for communicating with the FastAPI server.

This module provides a clean interface for making API calls to the FastAPI server
(Task 1) for text cleaning functionality. It handles network errors, retries,
and provides comprehensive error reporting.
"""

import requests
import time
import logging
from typing import Optional, Dict, Any
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

from .config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """
    Custom exception class for API client errors.

    This allows us to distinguish between our API client errors
    and other types of exceptions in the calling code.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        """
        Initialize API client error.

        Args:
            message (str): Error message describing what went wrong
            status_code (int, optional): HTTP status code if available
            response_data (dict, optional): Response data from the API if available
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class FastAPIClient:
    """
    HTTP client for communicating with the FastAPI text cleaning service.

    This class encapsulates all the logic for making HTTP requests to our
    FastAPI server, including error handling, retries, and response processing.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        """
        Initialize the FastAPI client.

        Args:
            base_url (str, optional): Base URL for the FastAPI server.
                                    Defaults to Config.FASTAPI_BASE_URL.
            timeout (int, optional): Request timeout in seconds.
                                   Defaults to Config.REQUEST_TIMEOUT.
        """
        self.base_url = base_url or Config.FASTAPI_BASE_URL
        self.timeout = timeout or Config.REQUEST_TIMEOUT

        # Create a session for connection pooling and performance
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'DocumentCleaner/1.0'
        })

        logger.info(f"FastAPI client initialized with base URL: {self.base_url}")

    def health_check(self) -> bool:
        """
        Check if the FastAPI server is running and responsive.

        Returns:
            bool: True if server is healthy, False otherwise

        Example:
            >>> client = FastAPIClient()
            >>> if client.health_check():
            ...     print("Server is running!")
            ... else:
            ...     print("Server is not available")
        """
        try:
            health_url = f"{Config.FASTAPI_BASE_URL}/health"
            logger.debug(f"Checking server health at: {health_url}")

            response = self.session.get(health_url, timeout=5)  # Short timeout for health check
            response.raise_for_status()

            logger.info("Server health check passed")
            return True

        except Exception as e:
            logger.warning(f"Server health check failed: {e}")
            return False

    def clean_text(self, text: str) -> str:
        """
        Send text to the FastAPI server for cleaning.

        This is the main method for text cleaning. It handles the complete
        request/response cycle including error handling and retries.

        Args:
            text (str): Text chunk to be cleaned

        Returns:
            str: Cleaned text from the API

        Raises:
            APIClientError: If the API call fails after all retries

        Example:
            >>> client = FastAPIClient()
            >>> dirty_text = "Some text\\n\\nPage 5\\n\\nmore content"
            >>> clean_text = client.clean_text(dirty_text)
            >>> print(clean_text)  # "Some text more content"
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided to clean_text")
            return text

        # Log the request (truncated for readability)
        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.debug(f"Cleaning text chunk: '{text_preview}' (length: {len(text)} chars)")

        # Prepare the request payload
        payload = {"text": text}

        # Try the request with retries
        last_exception = None

        for attempt in range(Config.MAX_RETRIES + 1):  # +1 for the initial attempt
            try:
                return self._make_clean_text_request(payload, attempt)

            except APIClientError as e:
                last_exception = e

                # Don't retry on client errors (4xx status codes)
                if e.status_code and 400 <= e.status_code < 500:
                    logger.error(f"Client error (attempt {attempt + 1}): {e}")
                    break

                # Log and potentially retry on server errors
                if attempt < Config.MAX_RETRIES:
                    wait_time = Config.RETRY_DELAY * (attempt + 1)  # Progressive backoff
                    logger.warning(f"API call failed (attempt {attempt + 1}/{Config.MAX_RETRIES + 1}): {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed. Last error: {e}")

        # If we get here, all attempts failed
        raise last_exception or APIClientError("All API attempts failed for unknown reason")

    def _make_clean_text_request(self, payload: Dict[str, Any], attempt: int) -> str:
        """
        Make a single HTTP request to the clean-text endpoint.

        This method is separated from clean_text() to make the retry logic cleaner
        and to allow for easier testing.

        Args:
            payload (dict): JSON payload to send to the API
            attempt (int): Current attempt number (for logging)

        Returns:
            str: Cleaned text from the API response

        Raises:
            APIClientError: If the request fails
        """
        clean_text_url = f"{Config.FASTAPI_BASE_URL}/clean-text"

        try:
            logger.debug(f"Making request to {clean_text_url} (attempt {attempt + 1})")

            response = self.session.post(
                clean_text_url,
                json=payload,
                timeout=self.timeout
            )

            # Log response status
            logger.debug(f"Received response with status: {response.status_code}")

            # Raise exception for HTTP error status codes
            response.raise_for_status()

            # Parse JSON response
            try:
                response_data = response.json()
            except ValueError as e:
                raise APIClientError(
                    f"Invalid JSON response from server: {e}",
                    status_code=response.status_code
                )

            # Extract cleaned text from response
            if "cleaned_text" not in response_data:
                raise APIClientError(
                    "Missing 'cleaned_text' field in API response",
                    status_code=response.status_code,
                    response_data=response_data
                )

            cleaned_text = response_data["cleaned_text"]

            # Log success
            original_length = len(payload["text"])
            cleaned_length = len(cleaned_text)
            logger.debug(f"Text cleaning successful: {original_length} -> {cleaned_length} chars")

            return cleaned_text

        except ConnectionError as e:
            raise APIClientError(f"Cannot connect to FastAPI server at {self.base_url}: {e}")

        except Timeout as e:
            raise APIClientError(f"Request timeout after {self.timeout} seconds: {e}")

        except HTTPError as e:
            # Try to get error details from response
            error_details = "Unknown error"
            try:
                if hasattr(e.response, 'json'):
                    error_data = e.response.json()
                    error_details = error_data.get('detail', str(error_data))
            except:
                error_details = e.response.text if hasattr(e.response, 'text') else str(e)

            raise APIClientError(
                f"HTTP {e.response.status_code}: {error_details}",
                status_code=e.response.status_code
            )

        except RequestException as e:
            raise APIClientError(f"Request failed: {e}")

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the FastAPI server.

        This method calls the root endpoint ("/") to get basic server information.
        Useful for debugging and verification.

        Returns:
            dict or None: Server information if successful, None if failed
        """
        try:
            root_url = self.base_url + "/"
            response = self.session.get(root_url, timeout=5)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.warning(f"Could not get server info: {e}")
            return None

    def close(self):
        """
        Close the HTTP session and clean up resources.

        It's good practice to call this when you're done with the client,
        though it's not strictly necessary as the session will be cleaned
        up when the object is garbage collected.
        """
        if self.session:
            self.session.close()
            logger.debug("HTTP session closed")

    def __enter__(self):
        """Context manager entry. Allows using 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit. Automatically closes the session."""
        self.close()


def create_api_client() -> FastAPIClient:
    """
    Factory function to create a configured FastAPI client.

    This function provides a convenient way to create a client with
    default settings from the configuration.

    Returns:
        FastAPIClient: Configured API client instance

    Example:
        >>> client = create_api_client()
        >>> if client.health_check():
        ...     result = client.clean_text("Some dirty text")
        >>> client.close()
    """
    return FastAPIClient()


def test_api_connection() -> bool:
    """
    Test the connection to the FastAPI server.

    This is a convenience function for quickly testing if the server
    is accessible and responding correctly.

    Returns:
        bool: True if connection test passed, False otherwise
    """
    print("Testing FastAPI server connection...")

    try:
        with create_api_client() as client:
            # Test health check
            if not client.health_check():
                print("❌ Health check failed")
                return False

            # Test server info
            server_info = client.get_server_info()
            if server_info:
                print(f"✅ Server info: {server_info}")

            # Test actual text cleaning with a simple example
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
    # This block runs when the file is executed directly
    # Useful for testing the API client

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
