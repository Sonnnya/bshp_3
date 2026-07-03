import os
from pathlib import Path

from pydantic_settings import BaseSettings

VERSION = "3.2.4.4"
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    DB_URL: str = "mongodb://localhost:27017"
    SOURCE_FOLDER: Path = BASE_DIR / "data/source"
    TEMP_FOLDER: Path = BASE_DIR / "data/temp"
    MODEL_FOLDER: Path = BASE_DIR / "models"
    TEST_MODE: bool = False
    AUTH_SERVICE_URL: str = ""
    THREAD_COUNT: int = 4
    USED_RAM_LIMIT: float = 2e9
    USE_DETAILED_LOG: bool = True
    DATASET_BATCH_LENGTH: int = 0
    QUANTIZE: bool = False
    TASK_TYPE: str = "CPU"
    DEVICES: str | None = None
    METRICS_FOLDER: Path = BASE_DIR / "metrics"
    MAX_MODELS: int = 1

    class Config:
        env_file = "../../.env"
        extra = "allow"


settings = Settings()

for key, value in settings.__dict__.items():
    if not key.startswith("_"):
        globals()[key] = value
