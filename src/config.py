#App configuration (non-sensitive settings)
import os
from dotenv import load_dotenv
load_dotenv()

#db
DATABASE_URL = os.getenv("database_url")

#Groq API
groq_api_key = os.getenv("groq_api_key")
groq_model = os.getenv("groq_model")
max_tokens = 4096
temperature = 1

