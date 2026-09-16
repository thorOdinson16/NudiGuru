"""Filesystem paths for the backend, resolved relative to this package."""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"

DATA_DIR = APP_DIR / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
DTW_TEMPLATE_PATH = TEMPLATES_DIR / "dtw.json"
HUBERT_TEMPLATE_PATH = TEMPLATES_DIR / "hubert.json"

# Model weights and reference voices stay where they are (large, provided locally).
MODELS_DIR = BACKEND_DIR / "kn"
VOICES_DIR = BACKEND_DIR / "Voices"

STORAGE_DIR = BACKEND_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
TTS_CACHE_DIR = STORAGE_DIR / "tts_cache"


def ensure_storage_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
