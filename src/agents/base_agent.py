# base agent creation 
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from groq import Groq
from sqlalchemy.orm import Session

from config import groq_api_key,groq_model, max_tokens, temperature


logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """base for all the agnets."""
    def __init__(self):
        self.client = Groq(api_key=groq_api_key)
        self.model = groq_model
        self.memory: List[Dict[str, str]] = []  # Stores conversation history
    
    def _get_llm_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Obtient une réponse du modèle LLM.
        
        Args:
            prompt: Le prompt à envoyer au modèle
            system_message: Message système optionnel pour guider le modèle
            
        Returns:
            La réponse du modèle
        """
        try:
            messages = []
            
            # Added system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # adding the prompt of the user 
            messages.extend(self.memory)
            messages.append({"role": "user", "content": prompt})
            
            # API cal
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extraction et retour de la réponse
            llm_response = response.choices[0].message.content
             # Update memory
            self.memory.append({"role": "user", "content": prompt})
            self.memory.append({"role": "assistant", "content": llm_response})
            return llm_response
        
        except Exception as e:
            logger.error(f"Error calling Groq API {e}")
            return f"Sorry, an error occurred while communicating with the model {e}"
    
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