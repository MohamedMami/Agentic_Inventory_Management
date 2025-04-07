# base agent creation 
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from groq import Groq
from sqlalchemy.orm import Session

from config import groq_api_key,groq_model, max_tokens, temperature