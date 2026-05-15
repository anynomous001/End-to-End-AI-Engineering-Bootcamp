from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables into os.environ so LangSmith and others can access them
load_dotenv()

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    OPENROUTER_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Config()