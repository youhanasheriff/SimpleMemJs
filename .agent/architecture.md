# SimpleMem Architecture Deep Dive

## Overview

SimpleMem is a three-stage memory system designed for LLM agents, achieving:

- **43.24% F1 score** (26.4% better than Mem0, 75.6% better than LightMem)
- **~550 token cost** (30× fewer than full-context methods)
- **388.3s retrieval time** (32.7% faster than LightMem)

## Theoretical Foundations

### Complementary Learning Systems (CLS) Theory

SimpleMem is inspired by biological memory consolidation:

- **Episodic memory**: Rapid acquisition of specific experiences (atomic memory units)
- **Semantic memory**: Gradual abstraction of patterns (consolidated abstracts)
- **Sleep consolidation**: Background process integrating episodic to semantic (recursive consolidation)

### Information-Theoretic Motivation

Raw dialogues contain significant entropy redundancy:

- Social fillers ("um", "you know", "I see")
- Confirmation phrases ("okay", "sure", "got it")
- Repeated references to same entities

**Semantic Lossless Compression** preserves information content while reducing token count.

## Stage 1: Semantic Structured Compression

### Sliding Window Segmentation

```
Dialogue Stream: [t₀, t₁, t₂, t₃, t₄, t₅, t₆, ...]
                      ↓
Windows: [t₀-t₄], [t₂-t₆], [t₄-t₈], ...
         (overlapping with stride 2)
```

### Information Score Computation

For each window W_t, compute:

```
H(W_t) = α · |ℰ_new| + (1-α) · (1 - cos(E(W_t), E(H_prev)))
```

Where:

- `ℰ_new`: Set of new named entities in W_t not in previous history
- `E(·)`: Embedding function
- `H_prev`: Embedding of previous interaction context
- `α`: Balance between entity novelty and semantic divergence (default: 0.5)

### Redundancy Filtering

```
if H(W_t) < τ_redundant:
    skip(W_t)  # Do not process or store
else:
    process(W_t)  # Continue to normalization
```

### Context Normalization Pipeline

```
Raw Dialogue → Φ_extract → Φ_coref → Φ_time → Memory Units

Example:
"He'll meet her tomorrow at 2pm"
    ↓ Φ_coref
"Bob will meet Alice tomorrow at 2pm"
    ↓ Φ_time (reference: 2025-11-15T10:00:00)
"Bob will meet Alice on 2025-11-16T14:00:00"
```

### Memory Unit Schema

```typescript
interface MemoryUnit {
  id: string; // UUID v4
  content: string; // Atomic fact statement
  entities: string[]; // Named entities
  topic: string; // Category/topic
  timestamp: string; // ISO-8601
  salience: "high" | "medium" | "low";
  windowId: string; // Source window reference
  createdAt: string; // Processing timestamp
  embedding?: number[]; // Dense vector (added during indexing)
  lexicalTokens?: string[]; // BM25 tokens (added during indexing)
}
```

## Stage 2: Structured Indexing & Recursive Consolidation

### Multi-View Index Structure

```
Memory Bank 𝕄 = {
  semantic:  { v_k → m_k }   // Dense embedding index
  lexical:   { token → [m_k] }  // Inverted index for BM25
  symbolic:  { metadata → [m_k] }  // Structured filters
}
```

#### Semantic Layer

- Uses dense embeddings (e.g., OpenAI text-embedding-3-small, Qwen3-Embedding-0.6B)
- Enables fuzzy matching ("latte" matches "hot drink")
- Vector similarity search with cosine distance

#### Lexical Layer

- BM25-style sparse representation
- Exact keyword and entity matching
- Handles proper nouns that may be OOV for embeddings

#### Symbolic Layer

- Structured metadata queries
- Entity type filtering (`entity_type = 'PERSON'`)
- Temporal range filtering (`timestamp BETWEEN start AND end`)

### Affinity Score for Consolidation

For memory units m_i and m_j:

```
ω_ij = cos(v_i, v_j) + β · exp(-|t_i - t_j| / γ)
```

Where:

- First term: Semantic similarity
- Second term: Temporal proximity (decay with γ)
- β: Weight for temporal factor (default: 0.3)
- γ: Temporal decay constant in seconds (default: 86400 = 1 day)

### Clustering Algorithm

```python
def find_clusters(units, τ_cluster):
    clusters = []
    available = set(units)

    while available:
        seed = available.pop()
        cluster = {seed}
        candidates = available.copy()

        for candidate in candidates:
            if min(ω(c, candidate) for c in cluster) >= τ_cluster:
                cluster.add(candidate)
                available.remove(candidate)

        if len(cluster) >= MIN_CLUSTER_SIZE:
            clusters.append(cluster)

    return clusters
```

### Consolidation Process

```
Cluster {m₁, m₂, m₃}:
  - m₁: "User ordered latte at 2025-11-15T08:00:00"
  - m₂: "User ordered cappuccino at 2025-11-16T08:30:00"
  - m₃: "User ordered latte at 2025-11-17T08:15:00"
        ↓
Abstract:
  pattern: "User regularly orders coffee beverages in the morning"
  sourceUnits: [m₁.id, m₂.id, m₃.id]
  frequency: 3
  lastOccurrence: "2025-11-17T08:15:00"
```

### Abstract Memory Schema

```typescript
interface AbstractMemory {
  id: string;
  pattern: string; // Consolidated pattern description
  sourceUnits: string[]; // IDs of contributing units
  frequency: number; // Occurrence count
  firstOccurrence: string; // Earliest timestamp
  lastOccurrence: string; // Most recent timestamp
  entities: string[]; // Union of entities from sources
  embedding?: number[]; // Pattern embedding
  isArchived: boolean; // If sources are archived
}
```

## Stage 3: Adaptive Query-Aware Retrieval

### Query Analysis

Before retrieval, analyze the query to determine:

1. **Complexity**: LOW or HIGH
2. **Lexical keywords**: Exact match terms
3. **Temporal constraints**: Time ranges
4. **Semantic query**: Rewritten for embedding matching

### Complexity Classification

| Feature           | LOW Complexity     | HIGH Complexity               |
| ----------------- | ------------------ | ----------------------------- |
| Query length      | Short (< 10 words) | Long (> 20 words)             |
| Time references   | Single/None        | Multiple/Comparative          |
| Aggregation words | None               | "all", "total", "how many"    |
| Comparison words  | None               | "compare", "difference", "vs" |
| Entity count      | Single             | Multiple                      |

### Dynamic Retrieval Depth

```
k_dyn = ⌊k_base · (1 + δ · C_q)⌋

Where:
- k_base = 3 (default base retrieval count)
- δ = 2.0 (expansion factor)
- C_q ∈ [0, 1] (query complexity score)

Examples:
- C_q = 0.0 (simple lookup): k_dyn = 3
- C_q = 0.5 (moderate): k_dyn = 6
- C_q = 1.0 (complex reasoning): k_dyn = 9
```

### Hybrid Scoring Function

```
S(q, m_k) = α · cos(E(q), v_k) + β · BM25(q, m_k) + 𝕀(constraints(q, m_k))

Where:
- α = 0.6 (semantic weight)
- β = 0.3 (lexical weight)
- 𝕀(·) = 1 if symbolic constraints satisfied, 0 otherwise
```

### Retrieval Strategy

```
LOW Complexity Query:
  1. Retrieve top-k_dyn abstract patterns matching semantically
  2. Return abstract summaries only (~100 tokens)

HIGH Complexity Query:
  1. Retrieve top-k_dyn memory units with highest hybrid scores
  2. For each unit, optionally expand with related abstracts
  3. Include temporal ordering for multi-event queries (~1000 tokens)
```

### Context Construction

```typescript
interface RetrievalContext {
  abstracts: AbstractMemory[]; // High-level patterns
  units: MemoryUnit[]; // Detailed factual entries
  totalTokens: number; // Estimated token count
  retrievalRationale: string; // Why these were selected
}
```

## Performance Optimizations

### Parallel Processing

```typescript
// Batch embedding generation
const embeddings = await provider.embed(texts.slice(0, BATCH_SIZE));

// Parallel retrieval across index layers
const [semantic, lexical, symbolic] = await Promise.all([
  semanticSearch(query),
  lexicalSearch(query),
  symbolicFilter(query),
]);
```

### Incremental Consolidation

Run consolidation as background task:

- Triggered when memory bank grows by N units
- Or periodically (e.g., every hour)
- Does not block main query/add operations

### Index Caching

```typescript
// LRU cache for frequent queries
const queryCache = new LRUCache<string, RetrievalContext>({
  max: 100,
  ttl: 1000 * 60 * 5, // 5 minutes
});
```

## Comparison with Other Systems

| System        | Memory Type | Compression  | Retrieval    | Token Efficiency |
| ------------- | ----------- | ------------ | ------------ | ---------------- |
| Full Context  | Raw         | None         | Full scan    | Very Low         |
| Mem0          | Structured  | Basic        | Vector       | Medium           |
| LightMem      | Lightweight | Summaries    | Vector       | Medium           |
| A-Mem         | Agentic     | Iterative    | Multi-step   | Low (slow)       |
| **SimpleMem** | **Atomic**  | **Semantic** | **Adaptive** | **High**         |

## References

1. Kumaran et al., 2016 - Complementary Learning Systems Theory
2. SimpleMem Paper - arXiv:2601.02553
3. Robertson & Zaragoza, 2009 - BM25 Scoring
