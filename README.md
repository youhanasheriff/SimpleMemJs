# SimpleMem

**Efficient Lifelong Memory for LLM Agents** - TypeScript Implementation

[![npm version](https://badge.fury.io/js/simplemem.svg)](https://www.npmjs.com/package/simplemem)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A TypeScript implementation of the [SimpleMem paper](https://arxiv.org/abs/2601.02553), achieving:

- **43.24% F1 score** (26.4% better than Mem0)
- **~550 token cost** (30× fewer than full-context methods)
- Cross-runtime support: **Bun**, **Node.js**, and **Deno**

## Installation

```bash
# Bun (recommended)
bun add @ys/simplemem

# npm
npm install @ys/simplemem

# pnpm
pnpm add @ys/simplemem
```

## Quick Start

```typescript
import { SimpleMem, OpenAIProvider, OpenAIEmbeddings } from "simplemem";

// Initialize with your provider
const memory = new SimpleMem({
  llm: new OpenAIProvider({
    apiKey: process.env.OPENAI_API_KEY!,
    // Optional: use OpenRouter or any OpenAI-compatible API
    // baseURL: 'https://openrouter.ai/api/v1',
  }),
  embeddings: new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY!,
  }),
});

// Add dialogues (Stage 1: Semantic Compression)
await memory.addDialogue(
  "Alice",
  "Let's meet at Starbucks tomorrow at 2pm",
  "2025-11-15T14:30:00",
);
await memory.addDialogue(
  "Bob",
  "Sure, I'll bring the market analysis report",
  "2025-11-15T14:31:00",
);

// Finalize processing
await memory.finalize();

// Query with adaptive retrieval (Stage 3)
const answer = await memory.ask("When and where will Alice and Bob meet?");
console.log(answer);
// Output: "16 November 2025 at 2:00 PM at Starbucks"
```

## Features

### Three-Stage Pipeline

1. **Semantic Structured Compression** (Stage 1)
   - Sliding window dialogue segmentation
   - Entropy-aware filtering
   - Coreference resolution (pronouns → names)
   - Temporal anchoring (relative → absolute timestamps)

2. **Structured Multi-View Indexing** (Stage 2)
   - **Semantic Layer**: Dense vector embeddings
   - **Lexical Layer**: BM25 keyword matching
   - **Symbolic Layer**: Metadata filtering (time, location, entities)

3. **Adaptive Query-Aware Retrieval** (Stage 3)
   - Query complexity estimation (LOW/HIGH)
   - Dynamic retrieval depth
   - Hybrid scoring across all layers
   - Optional reflection for complex queries

### Provider Flexibility

Use any OpenAI-compatible API:

```typescript
// OpenAI directly
new OpenAIProvider({ apiKey: "sk-..." });

// OpenRouter
new OpenAIProvider({
  apiKey: "sk-or-...",
  baseURL: "https://openrouter.ai/api/v1",
  model: "anthropic/claude-3-haiku",
});

// Local LLM (Ollama, LM Studio, etc.)
new OpenAIProvider({
  apiKey: "ollama",
  baseURL: "http://localhost:11434/v1",
  model: "llama3.2",
});
```

### Storage Options

```typescript
import { MemoryStorage, FileStorage } from 'simplemem';

// In-memory (default, ephemeral)
const memory = new SimpleMem({ storage: new MemoryStorage(), ... });

// File-based persistence
const memory = new SimpleMem({
  storage: new FileStorage({ path: './memory.json' }),
  ...
});
```

## API Reference

### SimpleMem

```typescript
class SimpleMem {
  constructor(options: SimpleMemOptions);

  // Add dialogue
  addDialogue(
    speaker: string,
    content: string,
    timestamp?: string,
  ): Promise<void>;
  addDialogues(
    dialogues: Array<{ speaker; content; timestamp }>,
  ): Promise<void>;
  finalize(): Promise<void>;

  // Query
  ask(question: string): Promise<string>;
  search(query: string, options?: SearchOptions): Promise<MemoryUnit[]>;
  getContext(query: string): Promise<RetrievalContext>;

  // Management
  getAllMemories(): Promise<MemoryUnit[]>;
  getMemoryCount(): Promise<number>;
  getStats(): { indexedUnits; bufferedDialogues; processedDialogues };
  export(): Promise<ExportData>;
  import(data: ExportData): Promise<void>;
  clear(): Promise<void>;
}
```

### Configuration

```typescript
const memory = new SimpleMem({
  llm: provider,
  embeddings: embeddings,
  storage: new MemoryStorage(),

  // Stage 1: Compression
  compression: {
    windowSize: 40, // Dialogues per window
    overlapSize: 2, // Window overlap
    redundancyThreshold: 0.3, // Filter low-info windows
  },

  // Stage 2: Indexing
  indexing: {
    semanticTopK: 25,
    keywordTopK: 5,
    semanticWeight: 0.6, // Hybrid score weight
    lexicalWeight: 0.3,
  },

  // Stage 3: Retrieval
  retrieval: {
    baseK: 3, // Base retrieval count
    complexityDelta: 2.0, // Expansion for complex queries
    enableReflection: true,
  },
});
```

## References

- **Paper**: [SimpleMem: Efficient Lifelong Memory for LLM Agents](https://arxiv.org/abs/2601.02553)
- **Original Python**: [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem)

## License

MIT
