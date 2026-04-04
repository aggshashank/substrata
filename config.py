"""Substrata configuration — loads settings from .env file."""

import functools
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "ollama/llama3.2:3b"
    EMBED_MODEL: str = "ollama/nomic-embed-text"
    WIKI_DIR: Path = Path("./wiki")
    SOURCES_DIR: Path = Path("./sources")
    CHROMA_DIR: Path = Path("./data/chroma")
    DB_PATH: Path = Path("./data/registry.db")
    MAX_LLM_CALLS_PER_RUN: int = 50
    CHUNK_SIZE: int = 750
    CHUNK_OVERLAP: int = 100
    LOG_LEVEL: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [
            self.WIKI_DIR,
            self.SOURCES_DIR,
            self.CHROMA_DIR,
            self.DB_PATH.parent,
            self.WIKI_DIR / "papers",
            self.WIKI_DIR / "concepts",
            self.WIKI_DIR / "daily",
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
