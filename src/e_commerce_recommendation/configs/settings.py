from typing import ClassVar
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project

class Settings(BaseSettings):
    """Application settings."""
    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    DATABASE_URL: str
    ASYNC_DATABASE_URL: str
    SAMPLE_DATABASE_URL: str
    IMAGE_ROOT: str
    HOST: str
    PORT: int
    SSL_CERT_FILE: str
    SSL_KEY_FILE: str

    @field_validator("SSL_CERT_FILE", "SSL_KEY_FILE")
    @classmethod
    def validate_file_exists(cls, v: str) -> str:
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(f"SSL file not found at: {path}")
        return str(path)


    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

if __name__ == "__main__":
    print("Settings Loaded:", settings.model_dump_json())