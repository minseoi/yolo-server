from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "YOLO Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model
    MODEL_PATH: str = "models/yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5  # API에서 override 가능

    # Debug Mode (서버 전역 설정)
    DEBUG_MODE: bool = False  # true면 모든 탐지에 bbox 이미지 저장

    # Paths
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    LOG_FILE: str = "logs/app.log"

    # Database (optional)
    DATABASE_URL: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"

    # Security
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
