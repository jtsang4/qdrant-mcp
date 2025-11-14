# Qdrant MCP Server

A Model Context Protocol (MCP) server that provides tools to store and retrieve information from a Qdrant vector database.

## Features

- **qdrant-store**: Store information with optional metadata in Qdrant collections
- **qdrant-find**: Retrieve relevant information using semantic search with vector name support
- **qdrant-debug**: Debug tool to inspect collection data structure and content
- Automatic collection creation
- OpenAI embedding generation for semantic search
- Configurable default collection
- Support for both dense and sparse vector search
- Built with TypeScript and modern tooling (Rslib, Biome, Vitest)

## Setup

Install the dependencies:

```bash
pnpm install
```

Build the project:

```bash
pnpm build
```

## Environment Variables

The server supports the following environment variables:

### Required

- `OPENAPI_API_KEY` or `OPENAI_API_KEY`: OpenAI API key for generating embeddings

### Optional

- `QDRANT_URL`: URL of the Qdrant server (default: `http://localhost:6333`)
- `QDRANT_API_KEY`: API key for Qdrant Cloud or authenticated instances
- `COLLECTION_NAME`: Default collection name to use
- `OPENAI_BASE_URL`: Custom OpenAI API base URL
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model to use (default: `text-embedding-3-small`)

## Available Tools

### qdrant-store

Store some information in the Qdrant database.

**Input:**

- `information` (string, required): Information to store
- `metadata` (object, optional): Optional metadata to store
- `collection_name` (string, optional): Name of the collection. Required if no default collection is configured.

**Returns:** Confirmation message with the stored information ID.

### qdrant-find

Retrieve relevant information from the Qdrant database using semantic search.

**Input:**

- `query` (string, required): Query to use for searching
- `vector_name` (string, optional): Name of the vector to use for searching. Available options: "dense" (recommended for semantic similarity), "sparse" (recommended for keyword matching). Default: "dense"
- `collection_name` (string, optional): Name of the collection to search in. Required if no default collection is configured.

**Returns:** Relevant information stored in the Qdrant database as separate messages, each with similarity scores.

### qdrant-debug

Debug tool to inspect collection data structure and content.

**Input:**

- `collection_name` (string, optional): Name of the collection to debug. Required if no default collection is configured.

**Returns:** Detailed information about the collection structure, configuration, and sample data points.

## Usage

### As a standalone MCP server

1. Start the server:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export QDRANT_URL="http://localhost:6333"
export COLLECTION_NAME="my-collection"

./dist/index.js
```

2. Configure your MCP client to connect to the server via stdio.

### With npx (Recommended)

You can run the Qdrant MCP server directly using npx without manual installation:

```bash
npx @jtsang/qdrant-mcp
```

For usage with environment variables:

```bash
OPENAI_API_KEY="your-openai-api-key" \
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
npx @jtsang/qdrant-mcp@latest
```

### With Claude Desktop

Add the following to your Claude Desktop configuration file:

#### Option 1: Using npx (Recommended)

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "npx",
      "args": ["-y", "@jtsang/qdrant-mcp@latest"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "my-collection"
      }
    }
  }
}
```

#### Option 2: Using local installation

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "node",
      "args": ["/path/to/qdrant-mcp/dist/index.js"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "my-collection"
      }
    }
  }
}
```

#### Option 3: Windows compatibility

For Windows users who encounter npx execution issues, use this configuration:

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@jtsang/qdrant-mcp@latest"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "my-collection"
      }
    }
  }
}
```

### Development

Run in development mode with auto-rebuild:

```bash
pnpm dev
```

Format code:

```bash
pnpm format
```

Check and fix code issues:

```bash
pnpm check
```

Run tests:

```bash
pnpm test
```

## Example Usage

### Store information

```typescript
// Using the qdrant-store tool
{
  "tool": "qdrant-store",
  "arguments": {
    "information": "The capital of France is Paris, known for the Eiffel Tower.",
    "metadata": {
      "category": "geography",
      "topic": "capitals"
    }
  }
}
```

### Find information

```typescript
// Using the qdrant-find tool
{
  "tool": "qdrant-find",
  "arguments": {
    "query": "What is the capital of France?",
    "vector_name": "dense"
  }
}
```

### Debug collection

```typescript
// Using the qdrant-debug tool
{
  "tool": "qdrant-debug",
  "arguments": {
    "collection_name": "my-collection"
  }
}
```

## Architecture

The server consists of:

- **QdrantHttpClient**: Handles HTTP API communication with Qdrant
- **QdrantMCPServer**: Main MCP server implementation with tool handlers
- **OpenAI Integration**: Generates embeddings for semantic search
- **TypeScript & Modern Tooling**: Built with Rslib for bundling, Biome for linting/formatting, and Vitest for testing

## Requirements

- Node.js 18+
- OpenAI API key
- Access to a Qdrant instance (local or cloud)

## License

MIT License - see the [LICENSE](LICENSE) file for details.
