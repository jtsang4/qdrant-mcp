# better-qdrant-mcp

An MCP server implemented with `fastmcp`, OpenAI embeddings, and `qdrant-client`, providing long-term memory and semantic search on top of Qdrant.

## User Guide

### Features

- **MCP server** built with `fastmcp`
- **Hybrid search** in Qdrant (dense OpenAI embeddings + sparse BM25)
- **Chinese support** via jieba
- **Three tools** aligned with the Node.js version:
  - `memory-store`
  - `memory-search`
  - `memory-debug`
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

Once the server is running, the MCP client will see three tools:

- `memory-store(information, metadata?: dict, collection_name?: str) -> str`
- `memory-search(query, limit?: int=5, collection_name?: str) -> str`
- `memory-debug(collection_name?: str) -> str`

`memory-search` uses hybrid search in Qdrant (dense + sparse). If the collection is configured with named vectors `dense` and `sparse`, queries are ranked by fusing dense OpenAI embeddings and sparse BM25 scores; otherwise it falls back to dense-only search.

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

```bash
# Start Qdrant + MCP
docker compose up -d

# Rebuild image and restart
docker compose up -d --build

# Stop services
docker compose down
```

## License

[MIT](LICENSE)
