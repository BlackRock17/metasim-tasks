"""
FastAPI server for text cleaning and chat functionality.

This server provides two main endpoints:
1. POST /clean-text - for cleaning text from artifacts using LLM
2. POST /chat - for sales chat conversation with AI buyer
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Import our custom modules
from .llm_service import get_llm_service
from .models import (
    CleanTextRequest,
    CleanTextResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Text Cleaning & Sales Chat API",
    description="API for cleaning text artifacts and conducting B2B sales conversations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Root endpoint - returns basic API information.
    """
    return {
        "message": "Text Cleaning & Sales Chat API is running",
        "version": "1.0.0",
        "endpoints": {
            "clean_text": "POST /clean-text",
            "chat": "POST /chat",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Tests LLM service connection.
    """
    try:
        llm_service = get_llm_service()
        connection_ok = llm_service.test_connection()

        return {
            "status": "healthy" if connection_ok else "unhealthy",
            "llm_service": "connected" if connection_ok else "disconnected",
            "timestamp": "2025-01-27T10:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )


@app.post("/clean-text", response_model=CleanTextResponse)
async def clean_text(request: CleanTextRequest):
    """
    Endpoint for cleaning text from artifacts.

    This endpoint removes headers, footers, page numbers and other noise from text
    using LLM-powered text cleaning. Perfect for processing extracted PDF content.

    Args:
        request (CleanTextRequest): Request containing text to be cleaned

    Returns:
        CleanTextResponse: Response with cleaned text

    Raises:
        HTTPException: If text cleaning fails or LLM service is unavailable
    """
    try:
        logger.info(f"Received text cleaning request. Length: {len(request.text)} characters")

        # Get LLM service instance
        llm_service = get_llm_service()

        # Clean the text using LLM
        cleaned_text = llm_service.clean_text(request.text)

        logger.info(f"Text cleaning completed. Result length: {len(cleaned_text)} characters")

        return CleanTextResponse(cleaned_text=cleaned_text)

    except Exception as e:
        logger.error(f"Error in clean_text endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text cleaning failed: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_conversation(request: ChatRequest):
    """
    Endpoint for B2B sales conversation with AI buyer.

    This endpoint simulates a conversation between a salesperson (user) and
    a skeptical B2B buyer (AI). The AI maintains conversation context and
    requires convincing arguments before agreeing to purchase.

    Args:
        request (ChatRequest): Request containing new message and chat history

    Returns:
        ChatResponse: Response with AI reply and updated conversation history

    Raises:
        HTTPException: If chat generation fails or LLM service is unavailable
    """
    try:
        logger.info(f"Received chat request. Message length: {len(request.message)} characters")
        logger.info(f"Chat history length: {len(request.chat_history)} messages")

        # Get LLM service instance
        llm_service = get_llm_service()

        # Convert ChatMessage objects to dict format for LLM service
        chat_history_dict = []
        for msg in request.chat_history:
            chat_history_dict.append({
                "role": msg.role,
                "content": msg.content
            })

        # Generate AI response using LLM
        ai_response = llm_service.chat_completion(
            message=request.message,
            chat_history=chat_history_dict
        )

        # Build updated conversation history
        updated_history = []

        # Add previous history
        for msg in request.chat_history:
            updated_history.append(msg)

        # Add user's new message
        updated_history.append(ChatMessage(
            role="user",
            content=request.message
        ))

        # Add AI response
        updated_history.append(ChatMessage(
            role="assistant",
            content=ai_response
        ))

        logger.info(f"Chat response generated. Response length: {len(ai_response)} characters")
        logger.info(f"Updated history length: {len(updated_history)} messages")

        return ChatResponse(
            response=ai_response,
            updated_history=updated_history
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation failed: {str(e)}"
        )


@app.post("/chat/clear")
async def clear_chat_history():
    """
    Endpoint for clearing chat conversation history.

    Useful for starting a new sales conversation from scratch.

    Returns:
        dict: Confirmation message
    """
    try:
        llm_service = get_llm_service()
        llm_service.clear_chat_history()

        logger.info("Chat history cleared successfully")

        return {
            "message": "Chat history cleared successfully",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear chat history: {str(e)}"
        )


if __name__ == "__main__":
    print("🚀 Starting Text Cleaning & Sales Chat API...")
    print("📝 Available endpoints:")
    print("   - http://localhost:8000/docs (API documentation)")
    print("   - POST http://localhost:8000/clean-text")
    print("   - POST http://localhost:8000/chat")
    print("   - POST http://localhost:8000/chat/clear")
    print("   - GET http://localhost:8000/health")

    uvicorn.run(
        "fastapi_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
