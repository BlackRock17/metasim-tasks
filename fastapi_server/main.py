"""
FastAPI server for text cleaning functionality.

Simple server with one endpoint for cleaning text from artifacts.
Task 1 from Metasim Interview Tasks - focused on text cleaning only.
"""

from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Text Cleaning API",
    description="Simple API for cleaning text from artifacts like headers, footers, page numbers",
    version="1.0.0"
)


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint - returns basic API information.
    """
    return {
        "message": "Text Cleaning API is running",
        "version": "1.0.0",
        "endpoint": "/clean-text"
    }


@app.post("/clean-text")
async def clean_text(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Endpoint for cleaning text from artifacts.

    Removes headers, footers, page numbers and other noise from text.

    Expected input:
    {
        "text": "Text to be cleaned"
    }

    Returns:
    {
        "cleaned_text": "Cleaned text without artifacts"
    }
    """
    try:
        # Check if 'text' field exists
        if "text" not in request:
            raise HTTPException(status_code=400, detail="Missing 'text' field in request")

        input_text = request["text"]

        # Check if text is valid
        if not input_text or not isinstance(input_text, str):
            raise HTTPException(status_code=400, detail="'text' field must be non-empty string")

        logger.info(f"Received text cleaning request. Length: {len(input_text)} characters")

        # TODO: Here we will integrate LLM Service for actual text cleaning
        # For now, return dummy cleaned text for testing
        cleaned_text = f"[CLEANED] {input_text}"

        logger.info(f"Text cleaning completed. Result length: {len(cleaned_text)} characters")

        return {
            "cleaned_text": cleaned_text
        }

    except HTTPException:
        # Re-raise HTTP exceptions (400, 500, etc.)
        raise
    except Exception as e:
        logger.error(f"Error in clean_text endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during text cleaning")


if __name__ == "__main__":
    # Start the server
    print("🚀 Starting Text Cleaning API...")
    print("📝 Available endpoints:")
    print("   - http://localhost:8000/docs (API documentation)")
    print("   - POST http://localhost:8000/clean-text")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
