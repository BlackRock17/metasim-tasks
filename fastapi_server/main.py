"""
Simple FastAPI server for text cleaning and chat functionality.

Provides two main endpoints:
1. POST /clean-text - for cleaning text artifacts using LLM
2. POST /chat - for sales chat conversation with AI buyer
"""

from fastapi import FastAPI, HTTPException
import uvicorn
import logging

# Import our modules
from .llm_service import get_llm_service
from .models import (
    CleanTextRequest,
    CleanTextResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Text Cleaning & Sales Chat API",
    description="API for cleaning text artifacts and conducting B2B sales conversations",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        llm_service = get_llm_service()
        connection_ok = llm_service.test_connection()

        return {
            "status": "healthy" if connection_ok else "unhealthy",
            "llm_service": "connected" if connection_ok else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.post("/clean-text", response_model=CleanTextResponse)
async def clean_text(request: CleanTextRequest):
    """
    Clean text by removing artifacts like headers, footers, page numbers.

    Perfect for processing extracted PDF content.
    """
    try:
        logger.info(f"Cleaning text: {len(request.text)} characters")

        llm_service = get_llm_service()
        cleaned_text = llm_service.clean_text(request.text)

        logger.info(f"Text cleaned: {len(cleaned_text)} characters")
        return CleanTextResponse(cleaned_text=cleaned_text)

    except Exception as e:
        logger.error(f"Text cleaning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text cleaning failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_conversation(request: ChatRequest):
    """
    B2B sales conversation with AI buyer.

    The AI acts as a skeptical B2B buyer who requires convincing arguments.
    """
    try:
        logger.info(f"Chat request: {len(request.message)} chars, {len(request.chat_history)} history")

        llm_service = get_llm_service()

        # Convert to dict format for LLM service
        chat_history_dict = []
        for msg in request.chat_history:
            chat_history_dict.append({
                "role": msg.role,
                "content": msg.content
            })

        # Generate AI response
        ai_response = llm_service.chat_completion(
            message=request.message,
            chat_history=chat_history_dict
        )

        # Build updated history
        updated_history = []

        # Add previous history
        for msg in request.chat_history:
            updated_history.append(msg)

        # Add new messages
        updated_history.append(ChatMessage(role="user", content=request.message))
        updated_history.append(ChatMessage(role="assistant", content=ai_response))

        logger.info(f"Chat response generated: {len(ai_response)} chars")
        return ChatResponse(response=ai_response, updated_history=updated_history)

    except Exception as e:
        logger.error(f"Chat generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Text Cleaning & Sales Chat API...")
    print("📝 Available endpoints:")
    print("   - http://localhost:8000/docs (API documentation)")
    print("   - POST http://localhost:8000/clean-text")
    print("   - POST http://localhost:8000/chat")
    print("   - GET http://localhost:8000/health")

    uvicorn.run(
        "fastapi_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
