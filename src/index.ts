/**
 * SimpleMem - Efficient Lifelong Memory for LLM Agents
 *
 * A TypeScript implementation of the SimpleMem paper:
 * "SimpleMem: Efficient Lifelong Memory for LLM Agents" (arXiv:2601.02553)
 *
 * @packageDocumentation
 */

// Main class
export {
  SimpleMem,
  createSimpleMem,
  type SimpleMemOptions,
} from "./SimpleMem.js";

// Type definitions
export * from "./types/index.js";

// Storage adapters
export { MemoryStorage } from "./storage/memory.js";
export { FileStorage, type FileStorageOptions } from "./storage/file.js";

// Embedding providers
export {
  OpenAIEmbeddings,
  type OpenAIEmbeddingsOptions,
} from "./embeddings/openai.js";

// LLM providers
export { OpenAIProvider, type OpenAIProviderOptions } from "./llm/openai.js";

// Stage components (for advanced usage)
export {
  MemoryBuilder,
  type CompressionConfig,
  DEFAULT_COMPRESSION_CONFIG,
} from "./stages/compression.js";

export {
  HybridIndex,
  type IndexingConfig,
  DEFAULT_INDEXING_CONFIG,
} from "./stages/indexing.js";

export {
  HybridRetriever,
  AnswerGenerator,
  type RetrievalConfig,
  DEFAULT_RETRIEVAL_CONFIG,
} from "./stages/retrieval.js";

// Utilities
export {
  detectRuntime,
  isServer,
  getRuntimeInfo,
  type Runtime,
  type RuntimeInfo,
} from "./utils/runtime.js";

export {
  cosineSimilarity,
  euclideanDistance,
  BM25Scorer,
  computeAffinityScore,
  computeDynamicK,
  computeHybridScore,
  type BM25Params,
} from "./utils/similarity.js";

export {
  dayjs,
  parseTimestamp,
  parseTimeRange,
  getTimeDiffSeconds,
  isWithinRange,
  formatTimestamp,
  now,
} from "./utils/temporal.js";
