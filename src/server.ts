#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';
import { z } from 'zod';
import { QdrantHttpClient } from './client';

// Vector name enum for type safety
const VectorNameEnum = z.enum(['dense', 'sparse']);
type VectorName = z.infer<typeof VectorNameEnum>;

// Main MCP Server
export class QdrantMCPServer {
  private server: McpServer;
  private qdrantClient: QdrantHttpClient;
  private openai: OpenAI;
  private defaultCollection?: string;

  constructor() {
    // Initialize configuration first
    const qdrantUrl = process.env.QDRANT_URL || 'http://localhost:6333';
    const qdrantApiKey = process.env.QDRANT_API_KEY;
    const defaultCollection = process.env.COLLECTION_NAME;

    const openaiApiKey =
      process.env.OPENAPI_API_KEY || process.env.OPENAI_API_KEY;
    const openaiBaseUrl = process.env.OPENAI_BASE_URL;

    if (!openaiApiKey) {
      throw new Error(
        'OpenAI API key is required (OPENAPI_API_KEY or OPENAI_API_KEY)',
      );
    }

    this.server = new McpServer({
      name: 'qdrant-mcp-server',
      version: '1.0.0',
    });

    this.qdrantClient = new QdrantHttpClient({
      url: qdrantUrl,
      apiKey: qdrantApiKey,
      defaultCollection: defaultCollection,
    });

    this.openai = new OpenAI({
      apiKey: openaiApiKey,
      baseURL: openaiBaseUrl,
    });

    this.defaultCollection = defaultCollection;

    this.setupToolHandlers();
  }

  private setupToolHandlers(): void {
    // Define the schema for qdrant-store tool
    const qdrantStoreSchema = z.object({
      information: z.string().describe('Information to store'),
      metadata: z
        .record(z.any())
        .optional()
        .describe('Optional metadata to store'),
      ...(this.defaultCollection
        ? {}
        : {
            collection_name: z
              .string()
              .describe('Name of the collection to store the information in'),
          }),
    });

    // Register the qdrant-store tool
    this.server.registerTool(
      'qdrant-store',
      {
        description: 'Store some information in the Qdrant database',
        inputSchema: qdrantStoreSchema,
      },
      async (args: any, extra: any) => {
        try {
          const storeArgs = {
            information: args.information,
            metadata: args.metadata,
            collection_name: this.defaultCollection || args.collection_name,
          };
          return await this.handleStore(storeArgs as any);
        } catch (error) {
          const result: CallToolResult = {
            content: [
              {
                type: 'text' as const,
                text: `Error: ${error instanceof Error ? error.message : String(error)}`,
              },
            ],
          };
          return result;
        }
      },
    );

    // Define the schema for qdrant-find tool
    const qdrantFindSchema = z.object({
      query: z.string().describe('Query to use for searching'),
      vector_name: VectorNameEnum.default('dense').describe(
        'Name of the vector to use for searching. Available options: "dense" (recommended for semantic similarity), "sparse" (recommended for keyword matching). Default: "dense"',
      ),
      ...(this.defaultCollection
        ? {}
        : {
            collection_name: z
              .string()
              .describe('Name of the collection to search in'),
          }),
    });

    // Register the qdrant-find tool
    this.server.registerTool(
      'qdrant-find',
      {
        description: 'Retrieve relevant information from the Qdrant database',
        inputSchema: qdrantFindSchema,
      },
      async (args: any, extra: any) => {
        try {
          const findArgs = {
            query: args.query,
            collection_name: this.defaultCollection || args.collection_name,
            vector_name: args.vector_name,
          };
          return await this.handleFind(findArgs as any);
        } catch (error) {
          const result: CallToolResult = {
            content: [
              {
                type: 'text' as const,
                text: `Error: ${error instanceof Error ? error.message : String(error)}`,
              },
            ],
          };
          return result;
        }
      },
    );

    // Define the schema for qdrant-debug tool
    const qdrantDebugSchema = z.object({
      ...(this.defaultCollection
        ? {}
        : {
            collection_name: z
              .string()
              .describe('Name of the collection to debug'),
          }),
    });

    // Register the qdrant-debug tool
    this.server.registerTool(
      'qdrant-debug',
      {
        description:
          'Debug tool to inspect collection data structure and content',
        inputSchema: qdrantDebugSchema,
      },
      async (args: any, extra: any) => {
        try {
          const debugArgs = {
            collection_name: this.defaultCollection || args.collection_name,
          };
          return await this.handleDebug(debugArgs as any);
        } catch (error) {
          const result: CallToolResult = {
            content: [
              {
                type: 'text' as const,
                text: `Error: ${error instanceof Error ? error.message : String(error)}`,
              },
            ],
          };
          return result;
        }
      },
    );
  }

  private async handleStore(args: {
    information: string;
    metadata?: any;
    collection_name?: string;
  }): Promise<CallToolResult> {
    const collectionName = args.collection_name || this.defaultCollection;
    if (!collectionName) {
      throw new Error('Collection name is required');
    }

    // Generate embedding for the information
    const embeddingResponse = await this.openai.embeddings.create({
      model: process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small',
      input: args.information,
    });

    const embedding = embeddingResponse.data[0].embedding;

    // Create collection if it doesn't exist
    await this.qdrantClient.createCollection(collectionName, embedding.length);

    // Generate a unique ID for the point
    const pointId = Date.now().toString();

    // Store the point with metadata
    const payload: any = {
      information: args.information,
      stored_at: new Date().toISOString(),
    };

    if (args.metadata) {
      payload.metadata = args.metadata;
    }

    await this.qdrantClient.upsertPoints(collectionName, [
      {
        id: pointId,
        vector: embedding,
        payload: payload,
      },
    ]);

    return {
      content: [
        {
          type: 'text' as const,
          text: `Information stored successfully in collection '${collectionName}' with ID: ${pointId}`,
        },
      ],
    };
  }

  private async handleFind(args: {
    query: string;
    collection_name?: string;
    vector_name?: VectorName;
  }): Promise<CallToolResult> {
    const collectionName = args.collection_name || this.defaultCollection;
    if (!collectionName) {
      throw new Error('Collection name is required');
    }

    // Generate embedding for the query
    const embeddingResponse = await this.openai.embeddings.create({
      model: process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small',
      input: args.query,
    });

    const queryVector = embeddingResponse.data[0].embedding;

    // Search for similar points with optional vector name
    const results = await this.qdrantClient.searchPoints(
      collectionName,
      queryVector,
      5, // default limit
      args.vector_name,
    );

    if (results.length === 0) {
      return {
        content: [
          {
            type: 'text' as const,
            text: `No relevant information found for query: "${args.query}"`,
          },
        ],
      };
    }

    // Format the results
    const formattedResults = results.map((result, index) => {
      const score = result.score || 0;
      const payload = result.payload || {};

      let response = `Result ${index + 1} (Score: ${score.toFixed(4)}):\n`;

      // Debug: Show payload structure for troubleshooting
      if (process.env.NODE_ENV === 'development' || process.env.DEBUG_QDRANT) {
        response += `Debug - Payload keys: ${Object.keys(payload).join(', ')}\n`;
      }

      response += `Detail: ${JSON.stringify(payload)}\n`;

      return response;
    });

    return {
      content: formattedResults.map((result) => ({
        type: 'text' as const,
        text: result,
      })),
    };
  }

  private async handleDebug(args: {
    collection_name?: string;
  }): Promise<CallToolResult> {
    const collectionName = args.collection_name || this.defaultCollection;
    if (!collectionName) {
      throw new Error('Collection name is required');
    }

    try {
      // Get collection info
      const collectionInfo = await this.qdrantClient.makeRequest(
        `/collections/${collectionName}`,
      );
      const info = (await collectionInfo.json()) as any;

      // Get some sample points
      const sampleResponse = await this.qdrantClient.makeRequest(
        `/collections/${collectionName}/points/scroll`,
        {
          method: 'POST',
          body: JSON.stringify({
            limit: 5,
            with_payload: true,
            with_vector: false,
          }),
        },
      );

      const sampleData = (await sampleResponse.json()) as any;

      let debugInfo = `Collection Info for "${collectionName}":\n`;
      debugInfo += JSON.stringify(info.result, null, 2);
      debugInfo += `\n\nSample Data (first ${sampleData.result?.result?.length || 0} points):\n`;

      if (sampleData.result?.result) {
        sampleData.result.result.forEach((point: any, index: number) => {
          debugInfo += `\n--- Point ${index + 1} (ID: ${point.id}) ---\n`;
          debugInfo += `Payload keys: ${Object.keys(point.payload || {}).join(', ')}\n`;
          debugInfo += `Payload: ${JSON.stringify(point.payload, null, 2)}\n`;
        });
      }

      return {
        content: [
          {
            type: 'text' as const,
            text: debugInfo,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text' as const,
            text: `Debug error: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Qdrant MCP Server running on stdio');
  }
}
