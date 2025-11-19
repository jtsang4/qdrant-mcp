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
                qm.PointStruct(
                    id=p["id"],
                    vector=p.get("vector"),
                    payload=p.get("payload"),
                )
                for p in points
            ],
        )

    def delete_points(self, name: str, ids: List[str]) -> None:
        # Delete one or more points by their IDs
        if not ids:
            return
        self._client.delete(
            collection_name=name,
            points_selector=qm.PointIdsList(points=ids),
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

    def hybrid_search(
        self,
        name: str,
        dense_vector: List[float],
        sparse_indices: Optional[List[int]] = None,
        sparse_values: Optional[List[float]] = None,
        limit: int = 5,
        fusion: qm.Fusion = qm.Fusion.RRF,
    ) -> List[Dict[str, Any]]:
        """Hybrid dense+sparse search using Qdrant query_points API.

        Falls back to dense-only search if sparse vectors are not provided.
        """

        if sparse_indices is None or sparse_values is None:
            return self.search(
                name=name, vector=dense_vector, limit=limit, vector_name="dense"
            )

        # Prefetch.query expects either a dense vector list or a sparse vector dict
        prefetch: List[qm.Prefetch] = [
            qm.Prefetch(query=dense_vector, using="dense"),
            qm.Prefetch(
                query={"indices": sparse_indices, "values": sparse_values},
                using="sparse",
            ),
        ]

        res = self._client.query_points(
            collection_name=name,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=fusion),
            limit=limit,
            with_payload=True,
        )

        out: List[Dict[str, Any]] = []
        # query_points returns a struct with .points list
        points = getattr(res, "points", res)
        for r in points:
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

    def retrieve_points(
        self, name: str, ids: List[str], with_vectors: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve specific points by their IDs with full details.

        Args:
            name: Collection name
            ids: List of point IDs to retrieve
            with_vectors: Whether to include vector data (default: True)

        Returns:
            List of point dictionaries containing id, payload, and optionally vectors
        """
        if not ids:
            return []

        points = self._client.retrieve(
            collection_name=name,
            ids=ids,
            with_payload=True,
            with_vectors=with_vectors,
        )

        out: List[Dict[str, Any]] = []
        for p in points:
            point_dict: Dict[str, Any] = {
                "id": p.id,
                "payload": p.payload or {},
            }
            if with_vectors and p.vector is not None:
                point_dict["vector"] = p.vector
            out.append(point_dict)
        return out

    def collection_info(self, name: str) -> Dict[str, Any]:
        info = self._client.get_collection(name)
        # qdrant-client returns a typed object; convert to dict
        return info.dict() if hasattr(info, "dict") else dict(info)  # type: ignore[arg-type]
