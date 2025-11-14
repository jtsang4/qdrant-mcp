// Configuration
interface QdrantConfig {
  url: string;
  apiKey?: string;
  defaultCollection?: string;
}

// Qdrant HTTP API Client
export class QdrantHttpClient {
  private config: QdrantConfig;

  constructor(config: QdrantConfig) {
    this.config = config;
  }

  public async makeRequest(
    endpoint: string,
    options: RequestInit = {},
  ): Promise<Response> {
    const url = `${this.config.url}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...((options.headers as Record<string, string>) || {}),
    };

    if (this.config.apiKey) {
      headers['api-key'] = this.config.apiKey;
    }

    return fetch(url, {
      ...options,
      headers,
    });
  }

  async createCollection(
    collectionName: string,
    vectorSize: number,
  ): Promise<void> {
    const response = await this.makeRequest(`/collections/${collectionName}`, {
      method: 'PUT',
      body: JSON.stringify({
        vectors: {
          size: vectorSize,
          distance: 'Cosine',
        },
      }),
    });

    if (!response.ok && response.status !== 409) {
      throw new Error(`Failed to create collection: ${response.statusText}`);
    }
  }

  async upsertPoints(collectionName: string, points: any[]): Promise<void> {
    const response = await this.makeRequest(
      `/collections/${collectionName}/points`,
      {
        method: 'PUT',
        body: JSON.stringify({
          points: points,
        }),
      },
    );

    if (!response.ok) {
      let errorDetails = response.statusText;
      try {
        const errorResponse = (await response.json()) as any;
        errorDetails =
          errorResponse.error?.message ||
          errorResponse.status?.error ||
          JSON.stringify(errorResponse);
      } catch {
        // If we can't parse the error response, use the status text
      }
      throw new Error(
        `Failed to upsert points in collection '${collectionName}': ${errorDetails}`,
      );
    }
  }

  async searchPoints(
    collectionName: string,
    queryVector: number[],
    limit: number = 5,
    vectorName?: string,
  ): Promise<any[]> {
    const requestBody: any = {
      limit: limit,
      with_payload: true,
    };

    // Handle named vectors vs default vectors
    if (vectorName) {
      requestBody.vector = {
        name: vectorName,
        vector: queryVector,
      };
    } else {
      requestBody.vector = queryVector;
    }

    const response = await this.makeRequest(
      `/collections/${collectionName}/points/search`,
      {
        method: 'POST',
        body: JSON.stringify(requestBody),
      },
    );

    if (!response.ok) {
      let errorDetails = response.statusText;
      try {
        const errorResponse = (await response.json()) as any;
        errorDetails =
          errorResponse.error?.message ||
          errorResponse.status?.error ||
          JSON.stringify(errorResponse);
      } catch {
        // If we can't parse the error response, use the status text
      }

      // If error mentions vector name requirement, suggest using 'dense' vector
      if (
        errorDetails.includes('requires specified vector name') &&
        !vectorName
      ) {
        errorDetails +=
          '. Hint: Try setting a vector name (e.g., "dense" or "sparse")';
      }

      throw new Error(
        `Failed to search points in collection '${collectionName}': ${errorDetails}. Request: ${JSON.stringify(requestBody)}`,
      );
    }

    const result = (await response.json()) as { result?: any[] };
    return result.result || [];
  }
}
