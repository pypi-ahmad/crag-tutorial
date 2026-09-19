"""Fixed tutorial configuration; credentials stay in the user environment."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
QDRANT_PATH = DATA_DIR / "qdrant"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
BASE_URL = DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
MODEL_NAME = "agnes-3.0-flash"
UPPER_THRESHOLD = 0.7
LOWER_THRESHOLD = 0.3
CACHE_VERSION = "crag-v2"

def check_environment():
    return {name: bool(os.environ.get(name)) for name in ("AGNESAI_API_KEY", "HF_TOKEN")}
