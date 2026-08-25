# app/config/settings.py
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://exam_user:exam_password@localhost:5432/exam_platform"
    database_pool_size: int = 20
    database_max_overflow: int = 40

    # Vector database
    vector_dimension: int = 1536

    # Storage
    storage_type: str = "local"  # local, s3
    storage_path: str = "./storage"
    s3_bucket: str = "exam-platform"
    s3_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # LLM / embedding providers (see app.services.llm / app.services.embeddings
    # for the provider abstraction — switching this value is the only change
    # needed to move between providers)
    llm_provider: str = "openai"  # openai, anthropic, gemini
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-5"
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"

    # Embeddings are a separate provider choice from llm_provider, since not
    # every LLM provider has a first-party embedding model (Anthropic doesn't;
    # OpenAI and Gemini both do). Defaults to openai for backward compatibility.
    embedding_provider: str = "openai"  # openai, gemini
    gemini_embedding_model: str = "gemini-embedding-001"
    # gemini-embedding-001 supports MRL truncation to an arbitrary
    # output_dimensionality — set to 1536 so it matches the existing pgvector
    # column (EMBEDDING_DIMENSION in app.database.models) with no migration.
    gemini_embedding_dimension: int = 1536

    # Telegram
    telegram_bot_token: str = "YOUR_BOT_TOKEN_HERE"
    telegram_webhook_url: Optional[str] = None

    # Admin auth
    admin_username: str = "admin"
    admin_password_hash: Optional[str] = None
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # Application
    app_name: str = "AI Exam Preparation Platform"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "your-secret-key-here-change-in-production"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"])

    # Worker — jobs run as FastAPI BackgroundTasks in-process (see
    # app.services.workers); sweep_secret authenticates the external cron
    # pinger that replaces Celery beat's periodic quiz-expiry sweep.
    sweep_secret: Optional[str] = None

    # Limits
    max_file_size: int = 104_857_600  # 100MB
    max_questions_per_quiz: int = 100
    max_quiz_duration_minutes: int = 180
    max_semantic_cache_entries: int = 10_000

    # LangGraph
    langgraph_checkpoint_store: str = "postgres"  # redis, postgres, memory
    langgraph_max_iterations: int = 50

    # Confidence thresholds for semantic resolution (spec sections 10, 12)
    exam_match_confidence_threshold: float = 0.7
    chapter_match_confidence_threshold: float = 0.7

    # Logging
    log_level: str = "INFO"
    log_json_format: bool = True


settings = Settings()
