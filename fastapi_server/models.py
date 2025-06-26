"""
Pydantic models for FastAPI request/response validation.

These models define the structure of data that the API accepts and returns.
Pydantic automatically validates input data and generates API documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# Text cleaning endpoint models
class CleanTextRequest(BaseModel):
    """
    Model for /clean-text endpoint request.

    Attributes:
        text (str): Text to be cleaned from artifacts like headers, footers, page numbers.
                   Minimum length 1 character, maximum 50000 characters.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Text to be cleaned from artifacts like headers, footers, page numbers",
        example="The purpose of discovery is to uncover your prospect's existing processes\\n\\nThe B2B Sales Process Handbook / 8\\n\\nand any pain points they may have."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Some text with artifacts\\n\\nPage 5\\n\\nmore content here"
            }
        }


class CleanTextResponse(BaseModel):
    """
    Model for /clean-text endpoint response.

    Attributes:
        cleaned_text (str): Text after cleaning artifacts.
    """
    cleaned_text: str = Field(
        ...,
        description="Text after cleaning artifacts",
        example="The purpose of discovery is to uncover your prospect's existing processes and any pain points they may have."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "cleaned_text": "Clean text without any artifacts"
            }
        }


# Chat endpoint models
class ChatMessage(BaseModel):
    """
    Model for a single chat message.

    Attributes:
        role (str): Role of the message sender - "user" or "assistant"
        content (str): Content of the message
    """
    role: str = Field(
        ...,
        pattern="^(user|assistant)$",
        description="Role of the message sender",
        example="user"
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Content of the message",
        example="Hello, I'm interested in your product"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, I would like to know more about your product"
            }
        }


class ChatRequest(BaseModel):
    """
    Model for /chat endpoint request.

    Attributes:
        message (str): New message from the user (salesperson)
        chat_history (List[ChatMessage]): Previous conversation history (optional)
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="New message from the user (salesperson)",
        example="I have a great product that can help your business grow"
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Previous conversation history",
        example=[]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, I have an amazing product for you!",
                "chat_history": [
                    {
                        "role": "assistant",
                        "content": "Hi there, what can I help you with?"
                    },
                    {
                        "role": "user",
                        "content": "I wanted to show you our new software"
                    }
                ]
            }
        }


class ChatResponse(BaseModel):
    """
    Model for /chat endpoint response.

    Attributes:
        response (str): Response from the AI buyer
        updated_history (List[ChatMessage]): Updated history including new messages
    """
    response: str = Field(
        ...,
        description="Response from the AI buyer",
        example="I'm not sure I need anything right now. What exactly does your product do?"
    )
    updated_history: List[ChatMessage] = Field(
        ...,
        description="Updated conversation history including new messages",
        example=[]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Interesting, but I'm quite busy right now. What makes your product special?",
                "updated_history": [
                    {
                        "role": "user",
                        "content": "Hello, I have an amazing product for you!"
                    },
                    {
                        "role": "assistant",
                        "content": "Interesting, but I'm quite busy right now. What makes your product special?"
                    }
                ]
            }
        }


# Error handling models
class ErrorResponse(BaseModel):
    """
    Standard model for error responses.

    Attributes:
        detail (str): Error message description
        error_code (str): Error code for programmatic handling (optional)
    """
    detail: str = Field(
        ...,
        description="Error message description",
        example="Invalid input data"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling",
        example="VALIDATION_ERROR"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Text field is required and cannot be empty",
                "error_code": "MISSING_TEXT"
            }
        }
