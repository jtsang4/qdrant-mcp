# better-qdrant-mcp

An MCP server implemented with `fastmcp`, OpenAI embeddings, and `qdrant-client`, providing long-term memory and semantic search on top of Qdrant.

## User Guide

### Features

- **MCP server** built with `fastmcp`
- **Hybrid search** in Qdrant (dense OpenAI embeddings + sparse BM25)
- **Chinese support** via jieba
- **Knowledge Base tools** (renamed from memory tools to avoid conflict):
  - `store-knowledge`
  - `store-knowledge-bulk`
  - `search-knowledge`
  - `get-knowledge-by-id`
  - `inspect-knowledge-base`
  - `delete-knowledge`
- **Multiple transports**: stdio, SSE, streamable HTTP

### Requirements

- Python 3.12+
- Qdrant reachable via HTTP

### Quick Start (published package)

The project is published as the `better-qdrant-mcp` package, so you can run it directly with `uvx` without cloning this repo.

#### 1. Environment variables (required for all transports)

Minimal env for typical use:

- `QDRANT_URL` – defaults to `http://localhost:6333`
- `QDRANT_API_KEY` – optional
- `COLLECTION_NAME` – optional default collection
- `OPENAI_API_KEY` (or `OPENAPI_API_KEY`) – required
- `OPENAI_BASE_URL` – optional
- `OPENAI_EMBEDDING_MODEL` – defaults to `text-embedding-3-small`

Advanced / transport-related env:

- `MCP_TRANSPORT` – `stdio` | `sse` | `streamable-http` (default: `stdio`)
- `MCP_HOST` – host for HTTP-based transports (default: `0.0.0.0`)
- `MCP_PORT` – port for HTTP-based transports (default: `8000`)
- `MCP_PATH` – path for HTTP transports (default: `/mcp`)

#### 2. Available MCP tools

Once the server is running, the MCP client will see these tools:

- `store-knowledge(content: str, title?: str, tags?: list[str], metadata?: dict, collection_name?: str) -> str`
- `store-knowledge-bulk(items: list[KnowledgeItem], collection_name?: str) -> str`
- `search-knowledge(query: str, limit?: int=5, collection_name?: str) -> str`
- `get-knowledge-by-id(ids: list[str] | str, collection_name?: str) -> str`
- `inspect-knowledge-base(collection_name?: str) -> str`
- `delete-knowledge(ids: list[str] | str, collection_name?: str) -> str`

**`store-knowledge`** automatically embeds the text using OpenAI and stores it in Qdrant (Knowledge Base), returning the stored point ID. The `title` and `tags` fields help improve search context and categorization.

**`store-knowledge-bulk`** efficiently stores multiple knowledge items at once using batch embedding. Each item in the list should include `content` (required), and optionally `title`, `tags`, and `metadata` fields. This is more efficient than calling `store-knowledge` multiple times.

**`search-knowledge`** uses hybrid search in Qdrant (dense + sparse). If the collection is configured with named vectors `dense` and `sparse`, queries are ranked by fusing dense OpenAI embeddings and sparse BM25 scores; otherwise it falls back to dense-only search.

**`get-knowledge-by-id`** retrieves the complete payload information for one or more knowledge items by their point IDs. Use this to inspect the full details of stored items (including `content`, `title`, `tags`, `metadata`, and `stored_at` timestamp). You can pass a single ID or a list of IDs (typically using the `id` field returned by `search-knowledge`).

**`inspect-knowledge-base`** shows the collection configuration and sample data points, useful for debugging and verification.

**`delete-knowledge`** deletes one or more stored knowledge items from Qdrant by their point IDs. You can pass a single ID or a list of IDs (typically using the `id` field returned by `search-knowledge`).

#### 3. Start the server

You can either specify the transport via CLI flags (recommended for quick start) or via env (`MCP_TRANSPORT`).

##### Standard IO (stdio) – default

```bash
uvx better-qdrant-mcp
```

In this mode, you configure your MCP client to use **stdio** transport and just invoke the binary; no HTTP URL is needed.

##### Server-Sent Events (SSE)

```bash
# Default host 0.0.0.0 and port 8000
uvx better-qdrant-mcp --transport sse

# Custom host and port
uvx better-qdrant-mcp --transport sse --host 0.0.0.0 --port 3000
```

Connection details for MCP clients:

- **Transport**: `sse`
- **URL**: `http://<host>:<port>/sse` (for example: `http://localhost:8000/sse`)

##### Streamable HTTP (recommended for web applications)

```bash
# Default host 0.0.0.0, port 8000 and path /mcp
uvx better-qdrant-mcp --transport streamable-http

# Custom host, port, and path
uvx better-qdrant-mcp --transport streamable-http --host 0.0.0.0 --port 3000 --path /api/mcp
```

Connection details for MCP clients:

- **Transport**: `streamable-http`
- **URL**: `http://<host>:<port><path>` (for example: `http://localhost:8000/mcp`)

## Development Guide

### Local installation (for development)

If you want to work on this repo locally instead of using the published package:

```bash
# using uv (recommended)
uv sync

# or with pip (editable install)
pip install -e .
```

### Local build

For local development, you can use the provided `Makefile`:

```bash
make build
```

This command will first clean the `dist` directory and then run `uv build` to produce fresh artifacts.

### Docker Deployment

Docker Compose provides Qdrant + this MCP server as a single service. Transport (`stdio`, `sse`, `streamable-http`) is selected via `MCP_TRANSPORT`.

The Docker image is built and published automatically to GitHub Container Registry as:

- `ghcr.io/jtsang4/better-qdrant-mcp:latest`
- Additional tags for branches, tags, and commit SHAs

The provided `docker-compose.yml` uses this published image directly, so you **do not need to build the image locally**.

```bash
# Start Qdrant + MCP using the published image
docker compose up -d

# Pull the latest published image and restart services
docker compose pull mcp && docker compose up -d

# Stop services
docker compose down
```

## License

[MIT](LICENSE)
