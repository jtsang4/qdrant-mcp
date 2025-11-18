from __future__ import annotations

from typing import Any, Dict, List, Annotated
import os
import argparse
from fastmcp import FastMCP
from pydantic import Field, BeforeValidator
import json

from .config import get_settings
from .embeddings import Embeddings, SparseEmbeddings
from .qdr_client import QdrClient
from .version import __version__


mcp = FastMCP("better-qdrant-mcp", version=__version__)
_qdr = QdrClient()
_settings = get_settings()


def _ensure_list(v: Any) -> List[str]:
    if isinstance(v, str):
        return [v]
    return v


@mcp.tool(
    name="store-knowledge",
    description=(
        "Store useful information or knowledge into the long-term knowledge base (Qdrant). "
        "This allows the agent to persist data that can be retrieved later or by other agents. "
        "Automatically embeds the text and returns the stored ID."
    ),
)
def store_knowledge(
    content: str = Field(
        ..., description="The content to store in the knowledge base."
    ),
    title: str = Field(
        "", description="Optional title for the content, helpful for search context."
    ),
    tags: List[str] = Field(
        default_factory=list, description="Optional list of tags for categorization."
    ),
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional JSON metadata to attach."
    ),
    collection_name: str = Field(
        "", description="Optional collection to use; defaults to env COLLECTION_NAME."
    ),
) -> str:
    """Store text into Qdrant with dense vectors produced by OpenAI embeddings.

    Returns:
        Confirmation message with the ID of the stored item.
    """
    collection = collection_name or _settings.default_collection
    if not collection:
        raise ValueError("Collection name is required")

    # Combine title and content for embedding to improve semantic search
    text_to_embed = f"{title}\n{content}" if title else content
    vector = Embeddings.embed_one(text_to_embed)

    # Ensure collection exists with the right dimensionality for dense vectors
    _qdr.ensure_collection(collection, len(vector))

    point_id = str(__import__("uuid").uuid4())
    payload: Dict[str, Any] = {
        "content": content,
        "stored_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    if title:
        payload["title"] = title
    if tags:
        payload["tags"] = tags
    if metadata:
        payload["metadata"] = metadata

    info = _qdr.collection_info(collection)
    vectors_cfg = info.get("config", {}).get("params", {}).get("vectors")
    has_named_dense = isinstance(vectors_cfg, dict) and "dense" in vectors_cfg

    # Try to detect sparse configuration
    sparse_cfg = info.get("config", {}).get("params", {}).get("sparse_vectors")
    has_sparse = isinstance(sparse_cfg, dict) and "sparse" in sparse_cfg

    point: Dict[str, Any] = {
        "id": point_id,
        "payload": payload,
    }

    if has_named_dense:
        # Named vectors: "dense" + optionally "sparse"
        vectors_dict = {"dense": vector}
        if has_sparse:
            indices, values = SparseEmbeddings.embed_one(text_to_embed)
            if indices and values:
                vectors_dict["sparse"] = {"indices": indices, "values": values}
        point["vector"] = vectors_dict
    else:
        # Legacy single vector
        point["vector"] = vector

    _qdr.upsert_points(collection, [point])

    return f"Information stored successfully in collection '{collection}' with ID: {point_id}"


@mcp.tool(
    name="search-knowledge",
    description=(
        "Search for relevant information in the long-term knowledge base using semantic search. "
        "Use this to retrieve context, facts, or past interactions stored in Qdrant."
    ),
)
def search_knowledge(
    query: str = Field(
        ..., description="The search query to find relevant information."
    ),
    limit: int = Field(
        5, description="Maximum number of results to return (default: 5)."
    ),
    collection_name: str = Field(
        "",
        description="Optional collection to target; defaults to env COLLECTION_NAME.",
    ),
) -> str:
    """Search for similar items in Qdrant using OpenAI embeddings for the query.

    Returns:
        JSON string containing a list of search results, each with 'score', 'point_id', and 'payload'.
    """
    collection = collection_name or _settings.default_collection
    if not collection:
        raise ValueError("Collection name is required")

    query_vec = Embeddings.embed_one(query)

    # Inspect collection config to decide whether hybrid search is available
    try:
        _info = _qdr.collection_info(collection)
    except Exception:
        _info = {}

    _vectors = (_info or {}).get("config", {}).get("params", {}).get("vectors")
    vectors_is_named = isinstance(_vectors, dict)
    has_named_dense = vectors_is_named and "dense" in _vectors

    sparse_cfg = (_info or {}).get("config", {}).get("params", {}).get("sparse_vectors")
    has_sparse = isinstance(sparse_cfg, dict) and "sparse" in sparse_cfg

    if has_named_dense and has_sparse:
        # Hybrid search: dense + sparse (BM25)
        indices, values = SparseEmbeddings.embed_one(query)
        results = _qdr.hybrid_search(
            collection,
            dense_vector=query_vec,
            sparse_indices=indices,
            sparse_values=values,
            limit=limit,
        )
    else:
        # Fallback: dense-only search (backward compatible)
        results = _qdr.search(
            collection,
            query_vec,
            limit,
            vector_name=("dense" if has_named_dense else None),
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
                "point_id": r.get("id"),
                "payload": r.get("payload", {}),
            }
        )

    return json.dumps(structured_results, ensure_ascii=False)


@mcp.tool(
    name="inspect-knowledge-base",
    description=(
        "Inspect the knowledge base configuration and view sample data. "
        "Useful for debugging collection settings or verifying stored content."
    ),
)
def inspect_knowledge_base(
    collection_name: str = Field(
        "",
        description="Optional collection to inspect; defaults to env COLLECTION_NAME.",
    ),
) -> str:
    """Inspect collection configuration and sample data.

    Returns:
        A formatted string containing collection info and sample data points.
    """
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


@mcp.tool(
    name="delete-knowledge",
    description=(
        "Delete specific information from the knowledge base using point IDs. "
        "Use the 'id' field returned from search-knowledge results."
    ),
)
def delete_knowledge(
    ids: Annotated[List[str], BeforeValidator(_ensure_list)] = Field(
        ...,
        description="A single point ID or a list of point IDs to delete. Use the 'id' from search-knowledge results.",
    ),
    collection_name: str = Field(
        "",
        description="Optional collection to target; defaults to env COLLECTION_NAME.",
    ),
) -> str:
    """Delete one or more memory points by their Qdrant IDs.

    Returns:
        Confirmation message of the number of deleted items.
    """
    collection = collection_name or _settings.default_collection
    if not collection:
        raise ValueError("Collection name is required")

    if not ids:
        raise ValueError("At least one ID is required to delete memory")

    # Input is already normalized to List[str] by validator
    _qdr.delete_points(collection, ids)

    return f"Deleted {len(ids)} item(s) from collection '{collection}'"


def run(
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8000,
    path: str = "/mcp",
) -> None:
    """
    Run the MCP server with specified transport.

    Args:
        transport: Transport type ("stdio", "sse", or "streamable-http")
        host: Host for HTTP-based transports (default: 0.0.0.0)
        port: Port for HTTP-based transports (default: 8000)
        path: Path for HTTP-based transports (default: /mcp)
    """
    if transport == "stdio":
        # Default stdio transport
        mcp.run(transport="stdio")
    elif transport == "sse":
        # Server-Sent Events transport
        mcp.run(transport="sse", host=host, port=port, path="/sse")
    elif transport == "streamable-http":
        # Streamable HTTP transport (newer standard)
        mcp.run(transport="streamable-http", host=host, port=port, path=path)
    else:
        raise ValueError(
            f"Unsupported transport: {transport}. Use 'stdio', 'sse', or 'streamable-http'"
        )


def main() -> None:
    """Entry point for command line interface with transport options."""
    parser = argparse.ArgumentParser(description="Better Qdrant MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type for MCP server (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP-based transports (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP-based transports (default: 8000)",
    )
    parser.add_argument(
        "--path", default="/mcp", help="Path for HTTP-based transports (default: /mcp)"
    )

    args = parser.parse_args()

    # Support environment variables as fallback
    transport = os.getenv("MCP_TRANSPORT", args.transport)
    host = os.getenv("MCP_HOST", args.host)
    port = int(os.getenv("MCP_PORT", str(args.port)))
    path = os.getenv("MCP_PATH", args.path)

    print(f"Starting Better Qdrant MCP Server with {transport} transport...")
    if transport != "stdio":
        print(
            f"Server will be available at http://{host}:{port}{path if transport == 'streamable-http' else '/sse'}"
        )

    run(transport=transport, host=host, port=port, path=path)
