from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Walk up from this file to find .env (works regardless of working directory)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    OPENROUTER_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Config()