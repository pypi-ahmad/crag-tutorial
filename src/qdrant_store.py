"""
Qdrant Vector Store Management for HotpotQA CRAG Tutorial.
Windows-native embedded Qdrant (CPU, free) with scikit-learn TF-IDF + TruncatedSVD 256-d embeddings.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from qdrant_client import QdrantClient, models
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CACHE_DIR, QDRANT_PATH
from src.cache import fingerprint, read_json, write_json

DEFAULT_COLLECTION = "hotpot_slice"
VECTOR_DIM = 256
ENCODER_CACHE_PATH = CACHE_DIR / "tfidf_svd_256.joblib"
INDEX_MANIFEST = CACHE_DIR / "qdrant_index_v2.json"


class TFIDFSVDEncoder:
    """
    CPU-only scikit-learn embedding pipeline combining TfidfVectorizer
    and TruncatedSVD to produce dense, unit-normalized 256-dimensional vectors.
    """

    def __init__(self, n_components: int = VECTOR_DIM, max_features: int = 15000):
        self.n_components = n_components
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.is_fitted = False

    def fit(self, texts: list[str]) -> "TFIDFSVDEncoder":
        tfidf_mat = self.tfidf.fit_transform(texts)
        actual_components = min(self.n_components, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        if actual_components < 1:
            raise ValueError("Index needs at least two documents and two vocabulary features.")
        if actual_components < self.n_components:
            self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.svd.fit(tfidf_mat)
        self.is_fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TFIDFSVDEncoder must be fitted before transforming texts.")
        tfidf_mat = self.tfidf.transform(texts)
        dense = self.svd.transform(tfidf_mat)
        # Pad with zeros if actual components was lower than VECTOR_DIM
        if dense.shape[1] < self.n_components:
            pad_width = self.n_components - dense.shape[1]
            dense = np.pad(dense, ((0, 0), (0, pad_width)), mode="constant")
        # Normalize to unit length for cosine distance
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (dense / norms).astype(np.float32)

    def save(self, path: Path | str = ENCODER_CACHE_PATH) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: Path | str = ENCODER_CACHE_PATH) -> "TFIDFSVDEncoder":
        return joblib.load(path)


_client_instance: QdrantClient | None = None
_client_path: Path | None = None


def get_qdrant_client(path: Path | str = QDRANT_PATH) -> QdrantClient:
    """
    Returns a singleton QdrantClient instance for embedded disk operation.
    Registers an atexit hook to release file locks on Windows cleanly.
    """
    global _client_instance, _client_path
    target_path = Path(path).resolve()
    if _client_instance is not None and target_path != _client_path:
        raise ValueError("Close the existing Qdrant client before opening a different path.")
    if _client_instance is None:
        target_path.mkdir(parents=True, exist_ok=True)
        _client_instance = QdrantClient(path=str(target_path))
        _client_path = target_path
        atexit.register(close_qdrant_client)
    return _client_instance


def close_qdrant_client() -> None:
    """Safely closes the embedded Qdrant client to release Windows file locks."""
    global _client_instance, _client_path
    if _client_instance is not None:
        try:
            _client_instance.close()
        except Exception:
            pass
        _client_instance = None
        _client_path = None


def get_or_fit_encoder(texts: list[str], force_refit: bool = False) -> TFIDFSVDEncoder:
    """Loads cached TF-IDF + SVD encoder or fits a new one over provided texts."""
    if ENCODER_CACHE_PATH.exists() and not force_refit:
        try:
            encoder = TFIDFSVDEncoder.load(ENCODER_CACHE_PATH)
            return encoder
        except Exception:
            pass

    encoder = TFIDFSVDEncoder(n_components=VECTOR_DIM)
    encoder.fit(texts)
    encoder.save(ENCODER_CACHE_PATH)
    return encoder


def index_slice(
    records: list[dict[str, Any]],
    collection_name: str = DEFAULT_COLLECTION,
    client: QdrantClient | None = None,
    force: bool = False,
    batch_size: int = 250,
) -> int:
    """
    Indexes the 200-question HotpotQA slice into embedded Qdrant.
    Idempotent: skips upsert only when count, encoder and corpus provenance match.

    Payload stored:
        - question_id: str
        - title: str
        - text: str
        - is_gold: bool

    Returns:
        Total number of points in the collection.
    """
    if client is None:
        client = get_qdrant_client()

    # Calculate expected points across all records
    expected_points = sum(len(r["context_paragraphs"]) for r in records)
    signature = fingerprint(["tfidf-svd-v2", sklearn.__version__, VECTOR_DIM, collection_name, records])
    manifest = read_json(INDEX_MANIFEST)
    encoder_valid = False
    if ENCODER_CACHE_PATH.exists() and isinstance(manifest, dict) and manifest.get("signature") == signature:
        try:
            encoder_valid = getattr(TFIDFSVDEncoder.load(), "corpus_signature", None) == signature
        except Exception:
            pass

    # Idempotent check
    if client.collection_exists(collection_name) and not force:
        info = client.get_collection(collection_name)
        count = info.points_count or 0
        first = client.retrieve(collection_name, ids=[0], with_payload=True)
        stored_signature = (first[0].payload or {}).get("_index_signature") if first else None
        if count == expected_points and encoder_valid and stored_signature == signature and info.config.params.vectors.size == VECTOR_DIM:
            print(
                f"[*] Collection '{collection_name}' already indexed with {count} points. Skipping upsert."
            )
            return count

    # Collect all paragraph texts and metadata
    all_texts: list[str] = []
    points_metadata: list[dict[str, Any]] = []

    point_id = 0
    for r in records:
        qid = r["id"]
        for p in r["context_paragraphs"]:
            all_texts.append(p["text"])
            points_metadata.append({
                "id": point_id,
                "question_id": qid,
                "title": p["title"],
                "text": p["text"],
                "is_gold": p["is_gold"],
            })
            point_id += 1

    # Fit / load encoder and produce embeddings
    encoder = get_or_fit_encoder(all_texts, force_refit=True)
    encoder.corpus_signature = signature
    encoder.save()
    vectors = encoder.transform(all_texts)

    # Validate/fit first so a bad input cannot erase the previous collection.
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
    )

    # Batch upsert
    total = len(points_metadata)
    print(f"[*] Upserting {total} points into collection '{collection_name}'...")
    for i in range(0, total, batch_size):
        batch_meta = points_metadata[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]

        points = [
            models.PointStruct(
                id=meta["id"],
                vector=batch_vectors[j].tolist(),
                payload={
                    "question_id": meta["question_id"],
                    "title": meta["title"],
                    "text": meta["text"],
                    "is_gold": meta["is_gold"],
                    "_index_signature": signature,
                },
            )
            for j, meta in enumerate(batch_meta)
        ]
        client.upsert(collection_name=collection_name, points=points)

    write_json(INDEX_MANIFEST, {"signature": signature, "count": total, "collection": collection_name})
    print(f"[OK] Successfully indexed {total} points into '{collection_name}'.")
    return total
