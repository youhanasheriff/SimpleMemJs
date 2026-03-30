# SimpleMem

**Efficient Lifelong Memory for LLM Agents** - TypeScript Implementation

[![npm version](https://badge.fury.io/js/@sheriax%2Fsimplemem.svg)](https://www.npmjs.com/package/@sheriax/simplemem)
[![CI](https://github.com/youhanasheriff/SimpleMemJs/actions/workflows/ci.yml/badge.svg)](https://github.com/youhanasheriff/SimpleMemJs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A TypeScript implementation of the [SimpleMem paper](https://arxiv.org/abs/2601.02553), achieving:

- **43.24% F1 score** (26.4% better than Mem0)
- **~550 token cost** (30x fewer than full-context methods)
- Cross-runtime support: **Bun**, **Node.js**, and **Deno**

## Installation

```bash
# Bun (recommended)
bun add @sheriax/simplemem

# npm
npm install @sheriax/simplemem

# pnpm
pnpm add @sheriax/simplemem
```

## Quick Start

```typescript
import { SimpleMem, OpenAIProvider, OpenAIEmbeddings } from "@sheriax/simplemem";

const memory = new SimpleMem({
  llm: new OpenAIProvider({
    apiKey: process.env.OPENAI_API_KEY!,
  }),
  embeddings: new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY!,
  }),
});

// Add dialogue
await memory.addDialogue("Alice", "Let's meet at Starbucks tomorrow at 2pm", "2025-11-15T14:30:00");
await memory.addDialogue("Bob", "Sure, I'll bring the report", "2025-11-15T14:31:00");
await memory.finalize();

// Query
const answer = await memory.ask("When and where will Alice and Bob meet?");
// => "16 November 2025 at 2:00 PM at Starbucks"
```

## Features

### Three-Stage Pipeline

1. **Semantic Structured Compression** (Stage 1)
   - Sliding window dialogue segmentation
   - Entropy-aware filtering
   - Coreference resolution (pronouns -> names)
   - Temporal anchoring (relative -> absolute timestamps)

2. **Structured Multi-View Indexing** (Stage 2)
   - **Semantic Layer**: Dense vector embeddings
   - **Lexical Layer**: BM25 keyword matching
   - **Symbolic Layer**: Metadata filtering (time, location, entities)

3. **Adaptive Query-Aware Retrieval** (Stage 3)
   - Query complexity estimation (LOW/HIGH)
   - Dynamic retrieval depth
   - Hybrid scoring across all layers
   - Optional reflection for complex queries

### Abstract Memory (Paper Section 3.2)

Automatically clusters similar memory units and generates higher-level patterns:

```typescript
// After adding many memories, trigger consolidation
const abstracts = await memory.consolidate();
// => [{ pattern: "Alice and Bob meet weekly at cafes to discuss project status", ... }]

// Abstracts are automatically included in retrieval context
const answer = await memory.ask("What are Alice's regular habits?");
```

Consolidation also runs automatically based on a configurable interval (default: every 50 new units).

### Flexible Input

Beyond dialogue, you can add raw text, documents, and direct facts:

```typescript
// Raw text (extracted via LLM)
await memory.addText("The quarterly review showed 15% revenue growth in Q3.");

// Long documents (automatically chunked)
await memory.addDocument(longArticle, {
  chunkSize: 2000,
  chunkOverlap: 200,
  source: "quarterly-report.pdf",
});

// Direct facts (no LLM extraction, stored immediately)
await memory.addFact("User prefers dark mode", {
  topic: "preferences",
  salience: "high",
});
```

### Event System

Subscribe to lifecycle events for observability, debugging, and custom workflows:

```typescript
memory.events.on("memory:units_created", ({ units, count }) => {
  console.log(`Created ${count} memory units`);
});

memory.events.on("retrieval:answer_generated", ({ query, answer }) => {
  telemetry.record({ query, answer });
});

memory.events.on("memory:abstract_created", ({ abstract }) => {
  console.log(`New pattern: ${abstract.pattern}`);
});

memory.events.on("error", ({ stage, error }) => {
  logger.error(`Error in ${stage}:`, error);
});
```

**Available events:** `memory:units_created`, `memory:units_indexed`, `memory:abstract_created`, `memory:consolidation_completed`, `retrieval:query_analyzed`, `retrieval:context_retrieved`, `retrieval:answer_generated`, `storage:cleared`, `error`

### Provider Flexibility

#### LLM Providers

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

#### Embedding Providers

```typescript
import { OpenAIEmbeddings, LocalEmbeddings, VoyageEmbeddings } from "@sheriax/simplemem";

// OpenAI (default)
new OpenAIEmbeddings({ apiKey: "sk-..." });

// Local embeddings (offline, free - requires @xenova/transformers)
new LocalEmbeddings({ model: "Xenova/all-MiniLM-L6-v2" });

// Multimodal embeddings (text + images)
new VoyageEmbeddings({ apiKey: "voyage-..." });
```

### Storage Options

```typescript
import { MemoryStorage, FileStorage, SQLiteStorage } from "@sheriax/simplemem";

// In-memory (default, ephemeral)
new SimpleMem({ storage: new MemoryStorage(), ... });

// File-based (JSON persistence)
new SimpleMem({ storage: new FileStorage({ path: "./memory.json" }), ... });

// SQLite (production-grade, ACID transactions, indexed queries)
new SimpleMem({ storage: new SQLiteStorage({ path: "./memory.db" }), ... });
```

SQLite uses `bun:sqlite` on Bun and `better-sqlite3` on Node.js (install as optional peer dep).

### Structured Logging

```typescript
import { silentLogger } from "@sheriax/simplemem";

// Default: logs warnings to console with [SimpleMem] prefix
new SimpleMem({ llm, embeddings });

// Silent: suppress all output
new SimpleMem({ llm, embeddings, logger: silentLogger });

// Custom logger
new SimpleMem({
  llm,
  embeddings,
  logger: {
    debug: (msg, ...args) => myLogger.debug(msg, ...args),
    info: (msg, ...args) => myLogger.info(msg, ...args),
    warn: (msg, ...args) => myLogger.warn(msg, ...args),
    error: (msg, ...args) => myLogger.error(msg, ...args),
  },
});
```

## API Reference

### SimpleMem

```typescript
class SimpleMem {
  // Lifecycle events
  readonly events: SimpleMemEventEmitter;

  constructor(options: SimpleMemOptions);

  // Dialogue input
  addDialogue(speaker: string, content: string, timestamp?: string | Date): Promise<void>;
  addDialogues(dialogues: Array<{ speaker; content; timestamp? }>): Promise<void>;
  finalize(): Promise<void>;

  // Flexible input
  addText(text: string, metadata?: TextMetadata): Promise<MemoryUnit[]>;
  addDocument(content: string, options?: DocumentOptions): Promise<MemoryUnit[]>;
  addFact(content: string, metadata?: FactMetadata): Promise<MemoryUnit>;

  // Query
  ask(question: string): Promise<string>;
  search(query: string, options?: SearchOptions): Promise<MemoryUnit[]>;
  getContext(query: string): Promise<RetrievalContext>;

  // Abstract memory
  consolidate(): Promise<AbstractMemory[]>;

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
    windowSize: 40,
    overlapSize: 2,
    redundancyThreshold: 0.3,
  },

  // Stage 2: Indexing
  indexing: {
    semanticTopK: 25,
    keywordTopK: 5,
    semanticWeight: 0.6,
    lexicalWeight: 0.3,
  },

  // Stage 3: Retrieval
  retrieval: {
    baseK: 3,
    complexityDelta: 2.0,
    enableReflection: true,
  },

  // Abstract memory consolidation
  abstraction: {
    clusterThreshold: 0.75,
    minClusterSize: 2,
    maxClusterSize: 20,
    consolidationInterval: 50,
  },

  // Logging
  logger: silentLogger,
});
```

## Setup for Contributors

After cloning, configure the pre-commit hook:

```bash
git config core.hooksPath .githooks
```

This enforces CHANGELOG.md updates and version bumps on every commit.

## References

- **Paper**: [SimpleMem: Efficient Lifelong Memory for LLM Agents](https://arxiv.org/abs/2601.02553)
- **Original Python**: [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem)

## License

MIT
