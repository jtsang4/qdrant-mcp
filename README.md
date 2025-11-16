# better-qdrant-mcp

A MCP server implemented with fastmcp, OpenAI embeddings, and qdrant-client. It provides three tools equivalent to the Node.js version:

- memory-store
- memory-search
- memory-debug

## Requirements

- Python 3.12+
- Qdrant reachable via HTTP

## Install deps (using uv or pip)

```bash
# using uv (recommended)
uv sync
# or with pip
pip install -e .
```

## Environment variables

- QDRANT_URL (default: http://localhost:6333)
- QDRANT_API_KEY (optional)
- COLLECTION_NAME (optional default collection)
- OPENAPI_API_KEY or OPENAI_API_KEY (required)
- OPENAI_BASE_URL (optional)
- OPENAI_EMBEDDING_MODEL (default: text-embedding-3-small)

## Run (stdio)

```bash
uvx better-qdrant-mcp
```

You can still run it via Python directly if you prefer:

```bash
python -m better_qdrant_mcp
```

Configure in MCP clients as a stdio server. Example (cursor-like):

```json
{
  "mcpServers": {
    "qdrant-mcp-python": {
      "type": "stdio",
      "command": "uvx",
      "args": ["better-qdrant-mcp"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "long_term_memory"
      }
    }
  }
}
```

## Tools

- memory-store(information, metadata?: dict, collection_name?: str) -> str
- memory-search(query, limit?: int=5, collection_name?: str) -> str
- memory-debug(collection_name?: str) -> str
