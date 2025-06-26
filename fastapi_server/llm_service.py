"""
LLM Service using LangChain for Azure OpenAI integration.

This module handles communication with Azure OpenAI API using LangChain for:
1. Text cleaning (removing artifacts like headers, footers, page numbers)
2. Chat conversations (for sales chat functionality)

Uses: LangChain + Azure OpenAI (as specified in requirements)
"""

import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


class LangChainAzureOpenAIService:
    """
    Service class for Azure OpenAI API communication using LangChain.

    Handles both text cleaning and chat functionality using GPT-4o model
    through LangChain abstractions for better prompt management and memory.
    """

    def __init__(self):
        """Initialize LangChain Azure OpenAI client."""

        # Get configuration from environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://metasim-openai-service.openai.azure.com/")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

        # Validate required configuration
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

        # Initialize LangChain Azure OpenAI client
        try:
            self.llm = AzureChatOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                deployment_name=self.deployment_name,
                api_version=self.api_version,
                temperature=0.7,  # Default temperature, will override per use case
                max_tokens=4000
            )

            # Initialize text cleaning chain
            self._setup_text_cleaning_chain()

            # Initialize chat chain with memory
            self._setup_chat_chain()

            logger.info("LangChain Azure OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain Azure OpenAI client: {e}")
            raise

    def _setup_text_cleaning_chain(self):
        """Setup LangChain chain for text cleaning."""

        # Text cleaning prompt template
        cleaning_template = """You are a text cleaning specialist. Your job is to clean text by removing artifacts that appear when converting formatted documents to plain text.

                               REMOVE these artifacts:
                               - Headers and footers (company names, document titles at top/bottom)
                               - Page numbers (e.g., "Page 1", "/ 2", "The B2B Sales Process Handbook / 8")
                               - Navigation elements (breadcrumbs, table of contents references)
                               - Formatting artifacts (random characters, broken spacing)
                               - Repeated elements that don't belong to main content
                               - Line breaks that interrupt sentences unnecessarily
                                
                               KEEP the main content:
                               - Preserve all meaningful text and paragraphs
                               - Keep proper sentence structure
                               - Maintain logical flow between sentences
                               - Preserve quotes and important information
                                
                               IMPORTANT: 
                               - Join broken sentences that were split across lines
                               - Fix spacing issues but preserve paragraph breaks
                               - Return ONLY the cleaned text, no explanations
                               - If text is already clean, return it unchanged
                                
                               Clean this text:
                                
                               {text}"""

        # Create prompt template
        self.cleaning_prompt = PromptTemplate(
            input_variables=["text"],
            template=cleaning_template
        )

        # Create LLM chain for text cleaning with low temperature for consistency
        cleaning_llm = AzureChatOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            deployment_name=self.deployment_name,
            api_version=self.api_version,
            temperature=0.1,  # Low temperature for consistent cleaning
            max_tokens=4000
        )

        self.cleaning_chain = LLMChain(
            llm=cleaning_llm,
            prompt=self.cleaning_prompt,
            verbose=False
        )

    def _setup_chat_chain(self):
        """Setup LangChain chain for chat conversations with memory."""

        # Chat system prompt template
        chat_system_prompt = """You are a potential customer in a B2B sales conversation. You are skeptical and need convincing before making any purchase decisions.

                                BEHAVIOR:
                                - Don't agree to buy immediately - you need to be convinced
                                - Ask relevant questions about the product/service
                                - Express concerns about price, implementation, ROI
                                - Be professional but cautious
                                - Only agree to purchase if the salesperson provides compelling arguments
                                - Show interest but maintain healthy skepticism
                                - Remember previous parts of the conversation
                                
                                RESPOND naturally as a busy business professional who values their time and money."""

        # Create chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", chat_system_prompt),
            ("human", "{input}")
        ])

        # Create LLM for chat with higher temperature for natural conversation
        chat_llm = AzureChatOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            deployment_name=self.deployment_name,
            api_version=self.api_version,
            temperature=0.7,  # Higher temperature for natural conversation
            max_tokens=1000
        )

        # Create memory for conversation
        self.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create chat chain
        self.chat_chain = LLMChain(
            llm=chat_llm,
            prompt=self.chat_prompt,
            verbose=False
        )

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing artifacts using LangChain.

        Args:
            text (str): Input text to be cleaned

        Returns:
            str: Cleaned text without artifacts

        Raises:
            Exception: If LangChain call fails
        """

        if not text or not text.strip():
            return text

        try:
            # Use LangChain cleaning chain
            result = self.cleaning_chain.run(text=text)

            cleaned_text = result.strip()
            logger.info(
                f"Text cleaning completed via LangChain. Input length: {len(text)}, Output length: {len(cleaned_text)}")

            return cleaned_text

        except Exception as e:
            logger.error(f"Error in LangChain text cleaning: {e}")
            raise Exception(f"Failed to clean text via LangChain: {str(e)}")

    def chat_completion(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate chat completion using LangChain with memory.

        Args:
            message (str): User message
            chat_history (List[Dict], optional): Previous conversation history

        Returns:
            str: AI response message

        Raises:
            Exception: If LangChain call fails
        """

        try:
            # If chat_history is provided, restore it to memory
            if chat_history:
                self.chat_memory.clear()  # Clear existing memory
                for msg in chat_history:
                    if msg["role"] == "user":
                        self.chat_memory.chat_memory.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        self.chat_memory.chat_memory.add_ai_message(msg["content"])

            # Generate response using LangChain
            response = self.chat_chain.run(input=message)

            # Add current interaction to memory
            self.chat_memory.chat_memory.add_user_message(message)
            self.chat_memory.chat_memory.add_ai_message(response)

            logger.info(f"Chat completion generated via LangChain. Response length: {len(response)}")

            return response.strip()

        except Exception as e:
            logger.error(f"Error in LangChain chat completion: {e}")
            raise Exception(f"Failed to generate chat response via LangChain: {str(e)}")

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get current chat history from LangChain memory.

        Returns:
            List[Dict]: Chat history in format [{"role": "user|assistant", "content": "..."}]
        """

        history = []
        messages = self.chat_memory.chat_memory.messages

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})

        return history

    def clear_chat_history(self):
        """Clear chat memory."""
        self.chat_memory.clear()
        logger.info("Chat history cleared")

    def test_connection(self) -> bool:
        """
        Test LangChain Azure OpenAI connection with a simple query.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Simple test using LangChain
            test_messages = [HumanMessage(content="Hello, just testing the connection.")]
            response = self.llm(test_messages)

            if response and response.content:
                logger.info("LangChain Azure OpenAI connection test successful")
                return True
            else:
                logger.error("LangChain connection test failed - no response content")
                return False

        except Exception as e:
            logger.error(f"LangChain connection test failed: {e}")
            return False


# Global service instance (Singleton pattern)
_llm_service_instance: Optional[LangChainAzureOpenAIService] = None


def get_llm_service() -> LangChainAzureOpenAIService:
    """
    Get singleton instance of LangChain LLM service.

    Returns:
        LangChainAzureOpenAIService: Initialized service instance
    """
    global _llm_service_instance

    if _llm_service_instance is None:
        _llm_service_instance = LangChainAzureOpenAIService()

    return _llm_service_instance


# Convenience functions for direct usage
def clean_text(text: str) -> str:
    """Convenience function for text cleaning via LangChain."""
    service = get_llm_service()
    return service.clean_text(text)


def chat_completion(message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Convenience function for chat completion via LangChain."""
    service = get_llm_service()
    return service.chat_completion(message, chat_history)


def get_chat_history() -> List[Dict[str, str]]:
    """Convenience function to get chat history."""
    service = get_llm_service()
    return service.get_chat_history()


def clear_chat_history():
    """Convenience function to clear chat history."""
    service = get_llm_service()
    service.clear_chat_history()


def test_connection() -> bool:
    """Convenience function for connection testing via LangChain."""
    service = get_llm_service()
    return service.test_connection()