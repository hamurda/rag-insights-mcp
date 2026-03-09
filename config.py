import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "unanswered_questions.db")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-5-mini")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
    MIN_CLUSTER_SIZE: int = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
    MAX_CLUSTER_SIZE: int = int(os.getenv("MAX_CLUSTER_SIZE", "20"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

    @classmethod
    def validate(cls) -> None:
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env or environment.")
        if not 0 <= cls.SIMILARITY_THRESHOLD <= 1:
            raise ValueError(f"SIMILARITY_THRESHOLD must be 0-1, got {cls.SIMILARITY_THRESHOLD}")

    @classmethod
    def db_path(cls) -> Path:
        return Path(__file__).parent / cls.DATABASE_PATH