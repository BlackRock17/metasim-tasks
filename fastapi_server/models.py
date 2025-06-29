"""
Simple Pydantic models for FastAPI request/response validation.

Basic models without complex validation for interview clarity.
"""

from pydantic import BaseModel
from typing import List, Optional


# Text cleaning models
class CleanTextRequest(BaseModel):
    """Request model for text cleaning endpoint."""
    text: str


class CleanTextResponse(BaseModel):
    """Response model for text cleaning endpoint."""
    cleaned_text: str


# Chat models
class ChatMessage(BaseModel):
    """Single chat message model."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    chat_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    updated_history: List[ChatMessage]
