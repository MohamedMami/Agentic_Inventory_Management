from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional
from groq import Groq
from sqlalchemy.orm import Session
from src.config import groq_api_key, groq_model, max_tokens, temperature

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents."""
    def __init__(self):
        self.client = Groq(api_key=groq_api_key)
        self.model = groq_model
    
    def _get_llm_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        try:
            messages = []
            
            # Add system message (if provided)
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add current user prompt
            messages.append({"role": "user", "content": prompt.strip()})
            
            # API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            if "rate limit" in str(e).lower():
                return "Error: API rate limit reached. Try again later."
            return f"Unexpected error: {str(e)}"
    
    @abstractmethod
    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """Abstract method for query processing."""
        pass