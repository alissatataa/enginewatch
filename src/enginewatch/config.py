"""Application settings, loaded from environment variables prefixed with ``EW_``."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EW_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./enginewatch.db"
    rul_model_path: str = "models/rul_baseline.joblib"
    warning_rul: float = 60.0
    critical_rul: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
