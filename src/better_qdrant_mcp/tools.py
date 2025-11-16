from __future__ import annotations

from typing import Any, Dict, Optional
from fastmcp import FastMCP
import json

from .config import get_settings
from .embeddings import Embeddings
from .qdr_client import QdrClient


mcp = FastMCP("qdrant-mcp-python")
_qdr = QdrClient()
_settings = get_settings()


@mcp.tool(
    name="memory-store",
    description=(
        "Store long-term textual information in the underlying vector memory store. "
        "Automatically embeds the text with OpenAI and returns the stored ID."
    ),
)
def memory_store(
    information: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: Optional[str] = None,
) -> str:
    """Store text into Qdrant with dense vectors produced by OpenAI embeddings.

    information: The main text content to store as long-term memory
    metadata: Optional JSON metadata to attach
    collection_name: Optional collection to use; defaults to env COLLECTION_NAME
    """
    collection = collection_name or _settings.default_collection
    if not collection:
        raise ValueError("Collection name is required")

    vector = Embeddings.embed_one(information)

    # Ensure collection exists with the right dimensionality
    _qdr.ensure_collection(collection, len(vector))

    point_id = str(__import__("time").time_ns())
    payload: Dict[str, Any] = {
        "information": information,
        "stored_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    if metadata:
        payload["metadata"] = metadata

    _qdr.upsert_points(
        collection,
        [
            {
                "id": point_id,
                "vector": (
                    {"dense": vector}
                    if (
                        isinstance(
                            _qdr.collection_info(collection)
                            .get("config", {})
                            .get("params", {})
                            .get("vectors"),
                            dict,
                        )
                        and "dense"
                        in _qdr.collection_info(collection)
                        .get("config", {})
                        .get("params", {})
                        .get("vectors", {})
                    )
                    else vector
                ),
                "payload": payload,
            }
        ],
    )

    return f"Information stored successfully in collection '{collection}' with ID: {point_id}"


@mcp.tool(
    name="memory-search",
    description=(
        "Retrieve relevant information previously stored in vector memory using semantic (dense) search with OpenAI embeddings."
    ),
)
def memory_search(
    query: str,
    limit: int = 5,
    collection_name: Optional[str] = None,
) -> str:
    """Search for similar items in Qdrant using OpenAI embeddings for the query."""
    collection = collection_name or _settings.default_collection
    if not collection:
        raise ValueError("Collection name is required")

    query_vec = Embeddings.embed_one(query)
    # Only pass a vector name if the collection defines a named 'dense' vector
    try:
        _info = _qdr.collection_info(collection)
        _vectors = (_info or {}).get("config", {}).get("params", {}).get("vectors")
        use_dense = isinstance(_vectors, dict) and "dense" in _vectors
    except Exception:
        use_dense = False

    results = _qdr.search(
        collection,
        query_vec,
        limit,
        vector_name=("dense" if use_dense else None),
    )

    if not results:
        return f'No relevant information found for query: "{query}"'

    # Return only structured data as serialized text (JSON string)
    structured_results: list[Dict[str, Any]] = []
    for r in results:
        structured_results.append(
            {
                "score": r.get("score", 0.0),
                "id": r.get("id"),
                "payload": r.get("payload", {}),
            }
        )

    return json.dumps(structured_results, ensure_ascii=False)


@mcp.tool(
    name="memory-debug",
    description=(
        "Debug/inspection tool for memory collections. Shows collection info and sample payloads."
    ),
)
def memory_debug(
    collection_name: Optional[str] = None,
) -> str:
    collection = collection_name or _settings.default_collection
    if not collection:
        raise ValueError("Collection name is required")

    info = _qdr.collection_info(collection)
    samples = _qdr.scroll_samples(collection, limit=5)

    out = [
        f'Collection Info for "{collection}":',
        __import__("json").dumps(info, indent=2),
    ]
    out.append("")
    out.append(f"Sample Data (first {len(samples)} points):")
    for idx, p in enumerate(samples, start=1):
        out.append(f"\n--- Point {idx} (ID: {p.get('id')}) ---")
        payload = p.get("payload", {})
        out.append(f"Payload keys: {', '.join(payload.keys())}")
        out.append("Payload: " + __import__("json").dumps(payload, indent=2))

    return "\n".join(out)


def run() -> None:
    # Default transport is stdio in fastmcp; be explicit
    mcp.run(transport="stdio")
