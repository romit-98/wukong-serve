"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Wukong-serve.

    All values may be overridden via environment variables. See ``.env.example``
    for the full list. No secrets should ever be hard-coded — anything sensitive
    must flow through this class.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_default_model: str = Field(default="llama3.2:3b")
    ollama_timeout_seconds: float = Field(default=30.0)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")

    # Admin
    admin_secret: str = Field(default="changeme")

    # Rate limiting (requests per minute)
    rate_limit_free_rpm: int = Field(default=10)
    rate_limit_pro_rpm: int = Field(default=60)

    # Circuit breaker
    circuit_breaker_threshold: int = Field(default=5)
    circuit_breaker_timeout_seconds: int = Field(default=30)

    # Sessions
    session_ttl_seconds: int = Field(default=1800)
    session_max_turns: int = Field(default=10)

    # Server
    log_level: str = Field(default="INFO")

    # Redis key prefixes (kept here so they are easy to grep)
    key_prefix_api_key: str = "wukong:apikey:"
    key_prefix_rate_limit: str = "wukong:ratelimit:"
    key_prefix_session: str = "wukong:session:"
    key_active_model: str = "wukong:active_model"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached so the env file is only parsed once per process.
    """
    return Settings()
