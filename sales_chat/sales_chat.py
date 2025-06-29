"""
Simple console application for B2B sales chat conversation.

The user plays a salesperson trying to convince an AI buyer to purchase a product.
The AI acts as a skeptical B2B buyer who requires convincing arguments.
"""

import sys
import requests
import json
import os
from typing import List, Dict

# Import config for API URL
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import Config


class SalesChatApp:
    """
    Simple console application for sales conversations.

    Handles the conversation loop and API communication with FastAPI server.
    """

    def __init__(self):
        """Initialize the sales chat application."""
        self.api_url = f"{Config.FASTAPI_BASE_URL}/chat"
        self.chat_history = []
        print("B2B Sales Chat Application")
        print("=" * 50)
        print("You are a SALESPERSON trying to sell a product.")
        print("The AI is a SKEPTICAL BUYER who needs convincing.")
        print("Type 'Bye' to exit the conversation.")
        print("=" * 50)

    def start_conversation(self):
        """Start the sales conversation with buyer greeting."""
        # Buyer starts with greeting
        buyer_greeting = self._get_ai_response("Hello, I'm here to learn about your offering.")
        self._display_message("BUYER", buyer_greeting)

        # Add to history
        self.chat_history.extend([
            {"role": "user", "content": "Hello, I'm here to learn about your offering."},
            {"role": "assistant", "content": buyer_greeting}
        ])

        # Start conversation loop
        self._conversation_loop()

    def _conversation_loop(self):
        """Main conversation loop."""
        while True:
            try:
                # Get user input (salesperson)
                print("\n" + "-" * 30)
                user_input = input("YOU (Salesperson): ").strip()

                # Check for exit
                if user_input.lower() == "bye":
                    print("\nBUYER: Goodbye! Thanks for the conversation.")
                    print("Sales conversation ended.")
                    return  # Clean exit

                if not user_input:
                    print("Please enter a message or 'Bye' to exit.")
                    continue

                # Get AI response
                ai_response = self._get_ai_response(user_input)
                self._display_message("BUYER", ai_response)

                # Update history
                self.chat_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": ai_response}
                ])

            except KeyboardInterrupt:
                print("\nConversation interrupted by user.")
                return
            except Exception as e:
                print(f"Error: {e}")
                print("Please make sure the FastAPI server is running.")
                return

    def _get_ai_response(self, message: str) -> str:
        """Get AI response from FastAPI server."""
        payload = {
            "message": message,
            "chat_history": self.chat_history
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        response.raise_for_status()
        result = response.json()
        return result["response"]

    def _display_message(self, sender: str, message: str):
        """Display a chat message with nice formatting."""
        print(f"\n{sender}: {message}")


def main():
    """Main function to run the sales chat application."""
    try:
        # Check if FastAPI server is running
        health_url = f"{Config.FASTAPI_BASE_URL}/health"
        health_response = requests.get(health_url, timeout=5)
        health_response.raise_for_status()

        # Start the chat application
        app = SalesChatApp()
        app.start_conversation()

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to FastAPI server.")
        print("Please start the server first:")
        print("   python -m fastapi_server.main")

    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed - {e}")

    except KeyboardInterrupt:
        print("\nSales conversation interrupted by user.")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
