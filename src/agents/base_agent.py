from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from groq import Groq
from sqlalchemy.orm import Session

from src.config import groq_api_key, groq_model, max_tokens, temperature

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents."""
    def __init__(self):
        self.client = Groq(api_key=groq_api_key)
        self.model = groq_model
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output",
        )
    
    def _get_llm_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        try:
            messages = []
            
            # Add system message (if provided)
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Process existing memory messages
            if self.memory.chat_memory.messages:
                for msg in self.memory.chat_memory.messages:
                    if isinstance(msg, BaseMessage):
                        role_mapping = {
                            "human": "user",
                            "ai": "assistant",
                            "system": "system",
                            "function": "function"  # Add if using functions
                        }
                        role = role_mapping.get(msg.type, msg.type)
                        content = msg.content.strip()
                        messages.append({
                            "role": role,
                            "content": content
                        })
                        logger.debug(f"Added message: {role}: {content[:50]}...")
                    else:
                        logger.warning(f"Ignoring invalid message: {msg}")
            
            # Add current user prompt
            messages.append({"role": "user", "content": prompt.strip()})
            
            # API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            llm_response = response.choices[0].message.content
            self.memory.chat_memory.add_user_message(prompt)
            self.memory.chat_memory.add_ai_message(llm_response)
            
            return llm_response
        
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            if "rate limit" in str(e).lower():
                return "Error: API rate limit reached. Try again later."
            return f"Unexpected error: {str(e)}"
    
    @abstractmethod
    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """Abstract method for query processing."""
        pass