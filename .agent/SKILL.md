---
name: SimpleMem TypeScript Implementation
description: Guidelines and architecture for implementing SimpleMem as a cross-runtime TypeScript/JavaScript npm package
---

# SimpleMem TypeScript Implementation Skill

This skill provides comprehensive guidance for implementing SimpleMem—an efficient lifelong memory system for LLM agents—as a TypeScript npm package compatible with Bun, Node.js, and Deno runtimes.

## Overview

SimpleMem achieves superior F1 scores (43.24%) with minimal token cost (~550 tokens) through a three-stage pipeline grounded in **Semantic Lossless Compression**.

## Architecture

### Stage 1: Semantic Structured Compression

Transform raw dialogue into atomic, self-contained memory units:

1. **Entropy-Aware Filtering**: Evaluate information density using a gating mechanism
   - Compute information score based on new entities and semantic novelty
   - Discard windows below `τ_redundant` threshold
2. **Context Normalization**:
   - **Coreference Resolution**: Replace pronouns with explicit entity names
   - **Temporal Anchoring**: Convert relative times to ISO-8601 timestamps

3. **Memory Unit Extraction**: Decompose utterances into minimal, indivisible factual statements

```typescript
interface MemoryUnit {
  content: string; // Self-contained factual statement
  entities: string[]; // Named entities mentioned
  topic: string; // High-level topic category
  timestamp: string; // ISO-8601 timestamp
  salience: "high" | "medium" | "low";
  embedding?: number[]; // Dense vector representation
}
```

### Stage 2: Structured Indexing & Recursive Consolidation

**Multi-View Indexing** across three layers:

1. **Semantic Layer**: Dense vector embeddings for fuzzy matching
2. **Lexical Layer**: Sparse representation for exact keyword/entity matches
3. **Symbolic Layer**: Structured metadata (timestamps, entity types) for deterministic filtering

**Recursive Consolidation**:

- Identify related units using affinity score: `ω_ij = cos(v_i, v_j) + β * exp(-|t_i - t_j|/γ)`
- When cluster affinity exceeds `τ_cluster`, consolidate into abstract representation
- Archive fine-grained entries while maintaining abstract patterns

```typescript
interface AbstractMemory {
  pattern: string; // Abstract pattern description
  sourceUnits: string[]; // IDs of consolidated units
  frequency: number; // How often this pattern occurs
  lastOccurrence: string; // Most recent timestamp
}
```

### Stage 3: Adaptive Query-Aware Retrieval

**Hybrid Scoring Function**:

```
S(q, m_k) = α * cos(E(q), v_k) + β * BM25(q, m_k) + 𝕀(constraints)
```

**Complexity-Aware Retrieval**:

- Estimate query complexity `C_q ∈ [0, 1]`
- Dynamic retrieval depth: `k_dyn = ⌊k_base * (1 + δ * C_q)⌋`
- Low complexity → Minimal abstract headers (~100 tokens)
- High complexity → Expanded atomic contexts (~1000 tokens)

## Key Hyperparameters

| Parameter             | Default | Description                                       |
| --------------------- | ------- | ------------------------------------------------- |
| `windowSize`          | 5       | Sliding window size for dialogue segmentation     |
| `redundancyThreshold` | 0.3     | τ_redundant for filtering low-information windows |
| `clusterThreshold`    | 0.75    | τ_cluster for triggering consolidation            |
| `temporalDecay`       | 0.1     | γ for temporal proximity weighting                |
| `semanticWeight`      | 0.6     | α for semantic similarity in hybrid scoring       |
| `lexicalWeight`       | 0.3     | β for lexical relevance                           |
| `baseRetrievalK`      | 3       | k_base for retrieval                              |
| `complexityDelta`     | 2.0     | δ for complexity-based expansion                  |

## System Prompts

### Stage 1: Semantic Structured Compression Prompt

```
You are a memory encoder in a long-term memory system. Your task is to transform raw conversational input into compact, self-contained memory units.

INPUT METADATA:
Window Start Time: {window_start_time} (ISO 8601)
Participants: {speakers_list}

INSTRUCTIONS:
1. Information Filtering:
   - Discard social filler, acknowledgements, and conversational routines that introduce no new factual or semantic information.
   - Discard redundant confirmations unless they modify or finalize a decision.
   - If no informative content is present, output an empty list.

2. Context Normalization:
   - Resolve all pronouns and implicit references into explicit entity names.
   - Ensure each memory unit is interpretable without access to prior dialogue.

3. Temporal Normalization:
   - Convert relative temporal expressions (e.g., "tomorrow", "last week") into absolute ISO 8601 timestamps using the window start time.

4. Memory Unit Extraction:
   - Decompose complex utterances into minimal, indivisible factual statements.

OUTPUT FORMAT (JSON):
{
  "memory_units": [
    {
      "content": "Alice agreed to meet Bob at the Starbucks on 5th Avenue on 2025-11-20T14:00:00.",
      "entities": ["Alice", "Bob", "Starbucks", "5th Avenue"],
      "topic": "Meeting Planning",
      "timestamp": "2025-11-20T14:00:00",
      "salience": "high"
    }
  ]
}
```

### Stage 2: Adaptive Retrieval Planning Prompt

```
Analyze the following user query and generate a retrieval plan. Your objective is to retrieve sufficient information while minimizing unnecessary context usage.

USER QUERY:
{user_query}

INSTRUCTIONS:
1. Query Complexity Estimation:
   - Assign "LOW" if the query can be answered via direct fact lookup or a single memory unit.
   - Assign "HIGH" if the query requires aggregation across multiple events, temporal comparison, or synthesis of patterns.

2. Retrieval Signals:
   - Lexical layer: extract exact keywords or entity names.
   - Temporal layer: infer absolute time ranges if relevant.
   - Semantic layer: rewrite the query into a declarative form suitable for semantic matching.

OUTPUT FORMAT (JSON):
{
  "complexity": "HIGH",
  "retrieval_rationale": "The query requires reasoning over multiple temporally separated events.",
  "lexical_keywords": ["Starbucks", "Bob"],
  "temporal_constraints": {
    "start": "2025-11-01T00:00:00",
    "end": "2025-11-30T23:59:59"
  },
  "semantic_query": "The user is asking about the scheduled meeting with Bob, including location and time."
}
```

### Stage 3: Reconstructive Synthesis Prompt

```
You are an assistant with access to a structured long-term memory.

USER QUERY:
{user_query}

RETRIEVED MEMORY (Ordered by Relevance):

[ABSTRACT REPRESENTATIONS]:
{retrieved_abstracts}

[DETAILED MEMORY UNITS]:
{retrieved_units}

INSTRUCTIONS:
1. Hierarchical Reasoning:
   - Use abstract representations to capture recurring patterns or general user preferences.
   - Use detailed memory units to ground the response with specific facts.

2. Conflict Handling:
   - If inconsistencies arise, prioritize the most recent memory unit.
   - Optionally reference abstract patterns when relevant.

3. Temporal Consistency:
   - Ensure all statements respect the timestamps provided in memory.

4. Faithfulness:
   - Base the answer strictly on the retrieved memory.
   - If required information is missing, respond with: "I do not have enough information in my memory."

FINAL ANSWER:
```

## TypeScript Implementation Guidelines

### Runtime Compatibility

The package must work seamlessly across:

- **Bun**: Native TypeScript support, fast runtime
- **Node.js**: Use `tsx` or compile to ESM/CJS
- **Deno**: Support `deno.json` imports

Use conditional imports and runtime detection:

```typescript
const runtime = detectRuntime(); // 'bun' | 'node' | 'deno'
```

### Storage Backends

Support pluggable storage adapters:

```typescript
interface StorageAdapter {
  save(key: string, data: unknown): Promise<void>;
  load<T>(key: string): Promise<T | null>;
  delete(key: string): Promise<void>;
  query<T>(filter: QueryFilter): Promise<T[]>;
}
```

Built-in adapters:

- **MemoryStorage**: In-memory (default, ephemeral)
- **FileStorage**: JSON files (persistent, no dependencies)
- **SQLiteStorage**: SQLite with better-sqlite3/bun:sqlite

### Embedding Providers

Support multiple embedding backends:

```typescript
interface EmbeddingProvider {
  embed(texts: string[]): Promise<number[][]>;
  dimensions: number;
}
```

Built-in providers:

- **OpenAIEmbeddings**: OpenAI API
- **TransformersEmbeddings**: Local transformers.js
- **OllamaEmbeddings**: Local Ollama

### LLM Providers

```typescript
interface LLMProvider {
  complete(prompt: string, options?: CompletionOptions): Promise<string>;
  stream?(prompt: string, options?: CompletionOptions): AsyncIterable<string>;
}
```

## Package Structure

```
simplemem/
├── src/
│   ├── index.ts                 # Main exports
│   ├── SimpleMem.ts             # Core SimpleMem class
│   ├── stages/
│   │   ├── compression.ts       # Stage 1: Semantic compression
│   │   ├── indexing.ts          # Stage 2: Multi-view indexing
│   │   ├── consolidation.ts     # Stage 2: Recursive consolidation
│   │   └── retrieval.ts         # Stage 3: Adaptive retrieval
│   ├── storage/
│   │   ├── adapter.ts           # Storage interface
│   │   ├── memory.ts            # In-memory storage
│   │   ├── file.ts              # File-based storage
│   │   └── sqlite.ts            # SQLite storage
│   ├── embeddings/
│   │   ├── provider.ts          # Embedding interface
│   │   ├── openai.ts            # OpenAI embeddings
│   │   └── transformers.ts      # Local transformers
│   ├── llm/
│   │   ├── provider.ts          # LLM interface
│   │   ├── openai.ts            # OpenAI/compatible
│   │   └── anthropic.ts         # Anthropic Claude
│   ├── utils/
│   │   ├── runtime.ts           # Runtime detection
│   │   ├── similarity.ts        # Cosine similarity, BM25
│   │   └── temporal.ts          # Time parsing utilities
│   └── types/
│       └── index.ts             # Type definitions
├── package.json
├── tsconfig.json
├── tsup.config.ts               # Build config for ESM/CJS
└── README.md
```

## API Design

```typescript
import { SimpleMem } from "simplemem";

// Initialize with configuration
const memory = new SimpleMem({
  llm: new OpenAIProvider({ apiKey: process.env.OPENAI_API_KEY }),
  embeddings: new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
  storage: new FileStorage({ path: "./memory.json" }),
  config: {
    windowSize: 5,
    redundancyThreshold: 0.3,
    baseRetrievalK: 3,
  },
});

// Add dialogue
await memory.addDialogue("Alice", "Let's meet tomorrow at 2pm", new Date());
await memory.addDialogue("Bob", "Sure, at Starbucks?", new Date());
await memory.finalize(); // Process pending dialogues

// Query with adaptive retrieval
const answer = await memory.ask("When is the meeting?");
console.log(answer);

// Low-level access
const units = await memory.search("Starbucks", { limit: 5 });
const abstracts = await memory.getAbstracts({ topic: "meetings" });
```

## References

- **Paper**: [SimpleMem: Efficient Lifelong Memory for LLM Agents](https://arxiv.org/abs/2601.02553)
- **GitHub**: [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem)
