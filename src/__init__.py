"""
Corrective RAG (CRAG) Package.
Windows-native tutorial implementation of arXiv:2401.15884.
"""

from src.config import (
    BASE_URL,
    MODEL_NAME,
    UPPER_THRESHOLD,
    LOWER_THRESHOLD,
    QDRANT_PATH,
    CACHE_DIR,
    DATA_DIR,
    check_environment,
)
from src.agnes_client import get_openai_client, chat
from src.data_hotpot import download_or_load, build_slice
from src.qdrant_store import index_slice, get_qdrant_client, close_qdrant_client
from src.retrieve import search
from src.crag import (
    evaluate_documents,
    decide_action,
    refine_strips,
    generate_answer,
    extract_json_object,
)
from src.comparison import run_naive_rag, run_comparison_50, automated_metrics, load_manual_reviews

__all__ = [
    "BASE_URL",
    "MODEL_NAME",
    "UPPER_THRESHOLD",
    "LOWER_THRESHOLD",
    "QDRANT_PATH",
    "CACHE_DIR",
    "DATA_DIR",
    "check_environment",
    "get_openai_client",
    "chat",
    "download_or_load",
    "build_slice",
    "index_slice",
    "get_qdrant_client",
    "close_qdrant_client",
    "search",
    "evaluate_documents",
    "decide_action",
    "refine_strips",
    "generate_answer",
    "extract_json_object",
    "run_naive_rag",
    "run_comparison_50",
    "automated_metrics",
    "load_manual_reviews",
]
