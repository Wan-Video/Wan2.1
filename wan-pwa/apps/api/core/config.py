from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API
    APP_NAME: str = "Wan2.1 PWA API"
    VERSION: str = "1.0.0"
    ENV: str = "development"
    PORT: int = 8000

    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # Replicate
    REPLICATE_API_TOKEN: str

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    # Redis (optional)
    REDIS_URL: str = "redis://localhost:6379"

    # Celery (optional)
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
