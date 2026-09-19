"""
Retrieval Module for HotpotQA CRAG Tutorial.
Provides vector search over embedded Qdrant with optional question_id pool filtering.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from src.qdrant_store import (
    DEFAULT_COLLECTION,
    ENCODER_CACHE_PATH,
    TFIDFSVDEncoder,
    get_qdrant_client,
)

_cached_encoder: TFIDFSVDEncoder | None = None
_encoder_stamp = None


def get_encoder() -> TFIDFSVDEncoder:
    """Retrieves or loads the cached TFIDFSVDEncoder."""
    global _cached_encoder, _encoder_stamp
    stamp = ENCODER_CACHE_PATH.stat().st_mtime_ns if ENCODER_CACHE_PATH.exists() else None
    if _cached_encoder is None or stamp != _encoder_stamp:
        if not ENCODER_CACHE_PATH.exists():
            raise RuntimeError(
                f"Encoder cache not found at {ENCODER_CACHE_PATH}. Run indexing first."
            )
        _cached_encoder = TFIDFSVDEncoder.load(ENCODER_CACHE_PATH)
        _encoder_stamp = stamp
    return _cached_encoder


def search(
    query: str,
    k: int = 5,
    question_id: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """
    Searches embedded Qdrant collection using TF-IDF + SVD dense vector query.

    Parameters:
        query: User question string.
        k: Maximum number of points to retrieve.
        question_id: Optional ID to restrict retrieval to that question's candidate pool (Main Search).
                     If None, performs open retrieval across the entire indexed corpus (Demo Search).
        collection_name: Qdrant collection name.
        client: Optional pre-configured QdrantClient.

    Returns:
        List of dicts:
            [
                {
                    "title": str,
                    "text": str,
                    "score": float,
                    "payload": dict
                },
                ...
            ]
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")
    if client is None:
        client = get_qdrant_client()

    encoder = get_encoder()
    first = client.retrieve(collection_name, ids=[0], with_payload=True)
    stored_signature = (first[0].payload or {}).get("_index_signature") if first else None
    if not stored_signature or stored_signature != getattr(encoder, "corpus_signature", None):
        raise RuntimeError("Collection and encoder provenance differ. Run index_slice before searching.")
    query_vector = encoder.transform([query])[0].tolist()

    # Build Qdrant filter if question_id is specified (Main Search)
    query_filter: models.Filter | None = None
    if question_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="question_id",
                    match=models.MatchValue(value=question_id),
                )
            ]
        )

    # Execute search via Qdrant query_points
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    )

    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        results.append({
            "title": payload.get("title", ""),
            "text": payload.get("text", ""),
            "score": float(point.score),
            "payload": payload,
        })

    return results
