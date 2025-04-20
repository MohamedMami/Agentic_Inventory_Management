# base agent creation 
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory  # New import
from langchain.schema import BaseMessage 
from groq import Groq
from sqlalchemy.orm import Session

from config import groq_api_key,groq_model, max_tokens, temperature


logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """base for all the agnets."""
    def __init__(self):
        self.client = Groq(api_key=groq_api_key)
        self.model = groq_model
        self.memory = ConversationBufferMemory()  # init langchain memory
    
    def _get_llm_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        try:
            messages = []

            # Add system message
            if system_message:
                messages.append({"role": "system", "content": system_message})

            # Convert memory to OpenAI-style format (MANUAL CONVERSION)
            if self.memory.chat_memory.messages:
                for msg in self.memory.chat_memory.messages:
                    # Extract role and content from the message object
                    if isinstance(msg, BaseMessage):
                        role_mapping = {
                            "human": "user",    # LangChain's "human" → Groq's "user"
                            "ai": "assistant",  # LangChain's "ai" → Groq's "assistant"
                            "system": "system"
                        }
                        role = role_mapping.get(msg.type, msg.type)  # Fallback to original if unknown
                        content = msg.content
                        messages.append({"role": role, "content": content})
                    else:
                        logger.warning(f"Invalid message format: {msg}")
            
            # Add current user's prompt
            messages.append({"role": "user", "content": prompt})

            # API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Extract response and update memory
            llm_response = response.choices[0].message.content
            self.memory.chat_memory.add_user_message(prompt)
            self.memory.chat_memory.add_ai_message(llm_response)

            return llm_response

        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"Sorry, an error occurred while communicating with the model: {e}"
    
    @abstractmethod
    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """
        Processes a user request.
        Args:
        query: The natural language query
        session: Database session
        Returns:
        Dictionary containing the response and associated data
        """
        pass