"""
Simple LLM Service using Azure OpenAI for text cleaning and chat functionality.

This module provides direct Azure OpenAI integration without complex LangChain abstractions.
Simplified for interview clarity while maintaining all required functionality.
"""

import os
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv

# LangChain imports - only what we need
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


class LLMService:
    """
    Simple LLM service for Azure OpenAI integration.

    Handles text cleaning and chat functionality with direct API calls.
    """

    def __init__(self):
        """Initialize Azure OpenAI client."""
        # Get configuration
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://metasim-openai-service.openai.azure.com/")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

        # Initialize Azure OpenAI client
        self.client = AzureChatOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            deployment_name=self.deployment_name,
            api_version=self.api_version,
            temperature=0.7,
            max_tokens=2000
        )

        # Simple chat history storage
        self.chat_history = []

        logger.info("LLM Service initialized successfully")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing artifacts.

        Args:
            text (str): Input text to be cleaned

        Returns:
            str: Cleaned text
        """
        if not text or not text.strip():
            return text

        # Simple cleaning prompt
        prompt = f"""Remove artifacts from this text like headers, footers, page numbers, and formatting noise.
                     Keep all meaningful content and fix broken sentences.
                     Return only the cleaned text without explanations.

                     Text to clean:
                     {text}"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.client(messages)
            cleaned = response.content.strip()

            logger.info(f"Text cleaned: {len(text)} → {len(cleaned)} chars")
            return cleaned

        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            raise Exception(f"Failed to clean text: {str(e)}")

    def chat_completion(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate chat response as skeptical B2B buyer.

        Args:
            message (str): User message
            chat_history (List[Dict], optional): Previous conversation

        Returns:
            str: AI response
        """
        try:
            # Build conversation messages
            messages = []

            # System message - define AI behavior
            system_prompt = """You are a BUYER, not a seller. You are meeting with a SALESPERSON who wants to sell YOU something.
                               You are skeptical and don't know what they're offering yet.
                               Keep responses concise (2-3 sentences max).
                               Start by asking the salesperson what THEY offer since you don't know their product.
                               Be professional but cautious. Ask questions about ROI, pricing, and value.
                               Don't agree to buy without compelling evidence."""

            messages.append(HumanMessage(content=f"System: {system_prompt}"))

            # Add chat history if provided
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))

            # Add current message
            messages.append(HumanMessage(content=message))

            # Generate response
            response = self.client(messages)
            reply = response.content.strip()

            logger.info(f"Chat response generated: {len(reply)} chars")
            return reply

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise Exception(f"Failed to generate chat response: {str(e)}")

    def test_connection(self) -> bool:
        """Test Azure OpenAI connection."""
        try:
            test_messages = [HumanMessage(content="Hello, test connection.")]
            response = self.client(test_messages)

            if response and response.content:
                logger.info("Azure OpenAI connection test successful")
                return True
            else:
                logger.error("Connection test failed - no response")
                return False

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Global service instance
_llm_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get singleton LLM service instance."""
    global _llm_service_instance

    if _llm_service_instance is None:
        _llm_service_instance = LLMService()

    return _llm_service_instance


# Convenience functions
def clean_text(text: str) -> str:
    """Convenience function for text cleaning."""
    service = get_llm_service()
    return service.clean_text(text)


def chat_completion(message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Convenience function for chat completion."""
    service = get_llm_service()
    return service.chat_completion(message, chat_history)


def test_connection() -> bool:
    """Convenience function for connection testing."""
    service = get_llm_service()
    return service.test_connection()
