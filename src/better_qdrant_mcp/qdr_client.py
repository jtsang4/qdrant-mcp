from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from .config import get_settings


class QdrClient:
    def __init__(self) -> None:
        s = get_settings()
        parsed = urlparse(s.qdrant_url)
        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "http":
            port = 80
        else:
            port = 443
        self._client = QdrantClient(
            prefer_grpc=False,
            port=port,
            url=s.qdrant_url,
            api_key=s.qdrant_api_key,
            timeout=s.qdrant_timeout,
        )

    def ensure_collection(self, name: str, vector_size: int) -> None:
        try:
            self._client.create_collection(
                collection_name=name,
                # Create as a named vector collection with 'dense' vector
                vectors_config={
                    "dense": qm.VectorParams(
                        size=vector_size, distance=qm.Distance.COSINE
                    )
                },
            )
        except Exception as e:  # collection may already exist
            # check exists; if exists, ignore, else re-raise
            try:
                info = self._client.get_collection(name)
                if info is None:
                    raise
            except Exception:
                raise e

    def upsert_points(
        self,
        name: str,
        points: List[Dict[str, Any]],
    ) -> None:
        # points: [{id, vector, payload}]
        self._client.upsert(
            collection_name=name,
            points=[
                qm.PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload"))
                for p in points
            ],
        )

    def search(
        self,
        name: str,
        vector: List[float],
        limit: int = 5,
        vector_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = self._client.search(
            collection_name=name,
            query_vector=vector
            if vector_name is None
            else qm.NamedVector(name=vector_name, vector=vector),
            limit=limit,
            with_payload=True,
        )
        out: List[Dict[str, Any]] = []
        for r in results:
            out.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
            )
        return out

    def scroll_samples(self, name: str, limit: int = 5) -> List[Dict[str, Any]]:
        points, _ = self._client.scroll(
            collection_name=name,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        return [{"id": p.id, "payload": p.payload} for p in points]

    def collection_info(self, name: str) -> Dict[str, Any]:
        info = self._client.get_collection(name)
        # qdrant-client returns a typed object; convert to dict
        return info.dict() if hasattr(info, "dict") else dict(info)  # type: ignore[arg-type]
