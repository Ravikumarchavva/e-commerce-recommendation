from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=True
    )

if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)
    print("Settings Loaded:", Settings().model_dump_json())