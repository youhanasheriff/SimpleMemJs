# SimpleMem TypeScript Implementation Plan

A comprehensive plan to implement SimpleMem as a cross-runtime TypeScript npm package compatible with Bun, Node.js, and Deno.

## Background

SimpleMem is an efficient lifelong memory system for LLM agents that achieves:

- **43.24% F1 score** with minimal token cost (~550 tokens)
- A three-stage pipeline: Semantic Compression → Structured Indexing → Adaptive Retrieval

## User Review Required

> [!IMPORTANT]
> **Embedding Provider Strategy**: Should we prioritize cloud APIs (OpenAI) or local models (transformers.js, Ollama) as the default? Local models add significant bundle size but work offline.

> [!IMPORTANT]
> **Storage Default**: For the default storage adapter, should we use:
>
> 1. In-memory only (simplest, no persistence)
> 2. File-based JSON (portable, no dependencies)
> 3. SQLite (best performance, requires native bindings)

---

## Proposed Changes

### Core Package Setup

#### [NEW] [package.json](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/package.json)

- Package name: `simplemem`
- Dual ESM/CJS exports for Node.js compatibility
- Bun-native TypeScript support
- Deno compatibility via `deno.json`

#### [NEW] [tsconfig.json](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/tsconfig.json)

- Target ES2022 for modern features
- Strict mode enabled
- Declaration files for types

#### [NEW] [tsup.config.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/tsup.config.ts)

- Build ESM and CJS bundles
- Generate declaration files
- Tree-shaking enabled

---

### Type Definitions

#### [NEW] [src/types/index.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/types/index.ts)

Core interfaces for:

- `MemoryUnit`: Atomic memory entries with content, entities, timestamp, embedding
- `AbstractMemory`: Consolidated patterns from recurring units
- `SimpleMemConfig`: Configuration options for the system
- `RetrievalContext`: Retrieved memories for query answering

---

### Stage 1: Semantic Structured Compression

#### [NEW] [src/stages/compression.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/stages/compression.ts)

- `DialogueWindow`: Sliding window over dialogue streams
- `computeInformationScore()`: Entropy-aware filtering using entity novelty + semantic divergence
- `extractMemoryUnits()`: LLM-based extraction with coreference resolution and temporal normalization
- Uses configurable `redundancyThreshold` (default: 0.3)

---

### Stage 2: Structured Indexing & Consolidation

#### [NEW] [src/stages/indexing.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/stages/indexing.ts)

- `SemanticIndex`: Dense vector storage with cosine similarity search
- `LexicalIndex`: BM25-style sparse index for exact keyword matching
- `SymbolicIndex`: Metadata filters (timestamp ranges, entity types)
- `HybridSearch`: Combines all three layers with configurable weights

#### [NEW] [src/stages/consolidation.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/stages/consolidation.ts)

- `computeAffinityScore()`: Semantic similarity + temporal proximity
- `findClusters()`: Clustering algorithm for related memory units
- `consolidateCluster()`: LLM-based pattern abstraction
- Background consolidation runner

---

### Stage 3: Adaptive Retrieval

#### [NEW] [src/stages/retrieval.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/stages/retrieval.ts)

- `analyzeQuery()`: LLM-based complexity estimation (LOW/HIGH)
- `computeRetrievalDepth()`: Dynamic k based on complexity
- `hybridScore()`: Combined semantic + lexical + symbolic scoring
- `constructContext()`: Build final context with abstracts and units

---

### Storage Adapters

#### [NEW] [src/storage/adapter.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/storage/adapter.ts)

```typescript
interface StorageAdapter {
  save(key: string, data: unknown): Promise<void>;
  load<T>(key: string): Promise<T | null>;
  delete(key: string): Promise<void>;
  query<T>(filter: QueryFilter): Promise<T[]>;
}
```

#### [NEW] [src/storage/memory.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/storage/memory.ts)

In-memory storage using Map (default, ephemeral)

#### [NEW] [src/storage/file.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/storage/file.ts)

JSON file-based persistence, portable across runtimes

#### [NEW] [src/storage/sqlite.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/storage/sqlite.ts)

SQLite storage using `bun:sqlite` or `better-sqlite3`

---

### Embedding Providers

#### [NEW] [src/embeddings/provider.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/embeddings/provider.ts)

```typescript
interface EmbeddingProvider {
  embed(texts: string[]): Promise<number[][]>;
  dimensions: number;
}
```

#### [NEW] [src/embeddings/openai.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/embeddings/openai.ts)

OpenAI text-embedding-3-small/large

#### [NEW] [src/embeddings/transformers.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/embeddings/transformers.ts)

Local embeddings via @xenova/transformers

---

### LLM Providers

#### [NEW] [src/llm/provider.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/llm/provider.ts)

```typescript
interface LLMProvider {
  complete(prompt: string, options?: CompletionOptions): Promise<string>;
  completeJSON<T>(prompt: string, schema: Schema): Promise<T>;
}
```

#### [NEW] [src/llm/openai.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/llm/openai.ts)

OpenAI GPT-4.1-mini compatible (works with Azure, Qwen)

#### [NEW] [src/llm/anthropic.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/llm/anthropic.ts)

Anthropic Claude support

---

### Main SimpleMem Class

#### [NEW] [src/SimpleMem.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/SimpleMem.ts)

Main orchestrator class:

```typescript
class SimpleMem {
  constructor(config: SimpleMemConfig);
  addDialogue(
    speaker: string,
    content: string,
    timestamp?: Date,
  ): Promise<void>;
  finalize(): Promise<void>; // Process pending dialogues
  ask(query: string): Promise<string>; // Full Q&A pipeline
  search(query: string, options?: SearchOptions): Promise<MemoryUnit[]>;
  getAbstracts(filter?: AbstractFilter): Promise<AbstractMemory[]>;
  consolidate(): Promise<void>; // Manual consolidation trigger
  export(): Promise<ExportData>;
  import(data: ExportData): Promise<void>;
}
```

---

### Utilities

#### [NEW] [src/utils/runtime.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/utils/runtime.ts)

Runtime detection for Bun/Node/Deno

#### [NEW] [src/utils/similarity.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/utils/similarity.ts)

- `cosineSimilarity()`: Vector similarity
- `bm25Score()`: Lexical relevance scoring

#### [NEW] [src/utils/temporal.ts](file:///Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs/src/utils/temporal.ts)

ISO-8601 parsing and temporal range utilities

---

## Verification Plan

### Automated Tests

Run with Bun (primary test runner):

```bash
cd /Users/youhanasheriff/Desktop/Sheriax/projects/community/SimpleMemJs
bun test
```

Tests to implement:

1. **Unit tests** for each stage:
   - `src/stages/__tests__/compression.test.ts`
   - `src/stages/__tests__/indexing.test.ts`
   - `src/stages/__tests__/retrieval.test.ts`

2. **Integration tests**:
   - `tests/integration/full-pipeline.test.ts` - End-to-end dialogue → query flow

3. **Storage adapter tests**:
   - `src/storage/__tests__/adapters.test.ts` - All adapters with same test suite

### Cross-Runtime Verification

```bash
# Bun
bun test

# Node.js (after build)
npm run build && node --experimental-vm-modules node_modules/jest/bin/jest.js

# Deno
deno test --allow-read --allow-write
```

### Manual Verification

1. **Basic Usage Test**:
   - Create a simple script using the API
   - Add 5-10 dialogues, run a query, verify response accuracy
2. **npm Publish Dry Run**:
   ```bash
   npm pack --dry-run
   ```
   Verify package contents include all necessary files

---

## Implementation Order

1. Package setup (package.json, tsconfig, build config)
2. Type definitions
3. Utility functions (similarity, temporal, runtime)
4. Storage adapters (memory → file → sqlite)
5. Embedding providers (OpenAI first)
6. LLM providers (OpenAI first)
7. Stage 1: Compression
8. Stage 2: Indexing + Consolidation
9. Stage 3: Retrieval
10. Main SimpleMem class
11. Tests
12. Documentation + Examples
