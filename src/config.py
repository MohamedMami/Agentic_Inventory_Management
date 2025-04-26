#App configuration (non-sensitive settings)
import os
from dotenv import load_dotenv
load_dotenv()

#db
username=os.getenv("username")
password= os.getenv("password")


#Groq API
groq_api_key = os.getenv("groq_api_key")
groq_model = os.getenv("groq_model")
max_tokens = 4096
temperature = 0.7

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")