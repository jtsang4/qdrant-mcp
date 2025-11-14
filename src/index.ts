#!/usr/bin/env node
import { QdrantMCPServer } from './server';

// Main execution
async function main(): Promise<void> {
  const server = new QdrantMCPServer();
  await server.run();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error('Server error:', error);
    process.exit(1);
  });
}
