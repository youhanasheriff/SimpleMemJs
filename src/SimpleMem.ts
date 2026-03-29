/**
 * SimpleMem - Efficient Lifelong Memory for LLM Agents
 *
 * Main orchestrator class integrating all three stages:
 * 1. Semantic Structured Compression (MemoryBuilder)
 * 2. Structured Indexing (HybridIndex)
 * 3. Adaptive Query-Aware Retrieval (HybridRetriever)
 */

import { v4 as uuidv4 } from "uuid";
import type {
  Dialogue,
  MemoryUnit,
  AbstractMemory,
  LLMProvider,
  EmbeddingProvider,
  StorageAdapter,
  ExportData,
  SearchOptions,
  RetrievalContext,
  Logger,
  TextMetadata,
  DocumentOptions,
  FactMetadata,
} from "./types/index.js";
import { consoleLogger } from "./types/index.js";
import { MemoryStorage } from "./storage/memory.js";
import {
  MemoryBuilder,
  type CompressionConfig,
  DEFAULT_COMPRESSION_CONFIG,
} from "./stages/compression.js";
import {
  HybridIndex,
  type IndexingConfig,
  DEFAULT_INDEXING_CONFIG,
} from "./stages/indexing.js";
import {
  HybridRetriever,
  AnswerGenerator,
  type RetrievalConfig,
  DEFAULT_RETRIEVAL_CONFIG,
} from "./stages/retrieval.js";
import {
  AbstractionEngine,
  type AbstractionConfig,
  DEFAULT_ABSTRACTION_CONFIG,
} from "./stages/abstraction.js";
import { now } from "./utils/temporal.js";
import { chunkText } from "./utils/chunking.js";
import { SimpleMemEventEmitter } from "./events.js";

// =============================================================================
// Configuration
// =============================================================================

export interface SimpleMemOptions {
  /**
   * LLM provider for extraction and retrieval
   */
  llm: LLMProvider;

  /**
   * Embedding provider for semantic search
   */
  embeddings: EmbeddingProvider;

  /**
   * Storage adapter (defaults to in-memory)
   */
  storage?: StorageAdapter;

  /**
   * Stage 1 configuration
   */
  compression?: Partial<CompressionConfig>;

  /**
   * Stage 2 configuration
   */
  indexing?: Partial<IndexingConfig>;

  /**
   * Stage 3 configuration
   */
  retrieval?: Partial<RetrievalConfig>;

  /**
   * Abstract memory consolidation configuration
   */
  abstraction?: Partial<AbstractionConfig>;

  /**
   * Logger instance (defaults to console logger)
   * Use `silentLogger` to suppress all output.
   */
  logger?: Logger;
}

// =============================================================================
// SimpleMem Class
// =============================================================================

/**
 * SimpleMem - Main Memory System
 *
 * Three-stage pipeline based on Semantic Lossless Compression:
 * 1. Semantic Structured Compression: addDialogue() → MemoryBuilder → storage
 * 2. Structured Indexing: HybridIndex with semantic/lexical/symbolic layers
 * 3. Adaptive Retrieval: ask() → HybridRetriever → AnswerGenerator
 */
export class SimpleMem {
  private llm: LLMProvider;
  private embeddings: EmbeddingProvider;
  private storage: StorageAdapter;
  private builder: MemoryBuilder;
  private index: HybridIndex;
  private retriever: HybridRetriever;
  private generator: AnswerGenerator;
  private abstractionEngine: AbstractionEngine;
  private logger: Logger;
  private dialogueCounter = 0;
  private initialized = false;

  /**
   * Event emitter for lifecycle hooks.
   * Subscribe to events like 'memory:units_created', 'retrieval:answer_generated', etc.
   */
  public readonly events: SimpleMemEventEmitter;

  constructor(options: SimpleMemOptions) {
    this.llm = options.llm;
    this.embeddings = options.embeddings;
    this.storage = options.storage ?? new MemoryStorage();
    this.logger = options.logger ?? consoleLogger;
    this.events = new SimpleMemEventEmitter();

    // Initialize Stage 1: Compression
    this.builder = new MemoryBuilder(
      this.llm,
      this.embeddings,
      options.compression ?? DEFAULT_COMPRESSION_CONFIG,
      this.logger,
    );

    // Initialize Stage 2: Indexing
    this.index = new HybridIndex(
      this.embeddings,
      options.indexing ?? DEFAULT_INDEXING_CONFIG,
    );

    // Initialize Stage 3: Retrieval
    this.retriever = new HybridRetriever(
      this.llm,
      this.index,
      options.retrieval ?? DEFAULT_RETRIEVAL_CONFIG,
      this.logger,
      this.storage,
    );

    this.generator = new AnswerGenerator(this.llm, this.logger);

    // Initialize Abstract Memory Consolidation
    this.abstractionEngine = new AbstractionEngine(
      this.llm,
      this.embeddings,
      this.storage,
      options.abstraction ?? DEFAULT_ABSTRACTION_CONFIG,
      this.logger,
    );
  }

  /**
   * Initialize from storage (load existing memories)
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    const units = await this.storage.getAllUnits();
    if (units.length > 0) {
      await this.index.addUnits(units);
      this.index.rebuildLexicalIndex();
    }

    this.initialized = true;
  }

  /**
   * Add a single dialogue
   *
   * @param speaker Speaker name
   * @param content Dialogue content
   * @param timestamp Timestamp (ISO 8601 format, defaults to now)
   */
  async addDialogue(
    speaker: string,
    content: string,
    timestamp?: string | Date,
  ): Promise<void> {
    const dialogue: Dialogue = {
      id: this.dialogueCounter++,
      speaker,
      content,
      timestamp:
        timestamp instanceof Date
          ? timestamp.toISOString()
          : (timestamp ?? now()),
    };

    this.builder.addDialogue(dialogue);

    // Process complete windows immediately
    if (this.builder.hasFullWindow()) {
      const units = await this.builder.processWindows();
      await this.saveAndIndexUnits(units);
    }
  }

  /**
   * Add multiple dialogues at once
   */
  async addDialogues(
    dialogues: Array<{
      speaker: string;
      content: string;
      timestamp?: string | Date;
    }>,
  ): Promise<void> {
    for (const d of dialogues) {
      const dialogue: Dialogue = {
        id: this.dialogueCounter++,
        speaker: d.speaker,
        content: d.content,
        timestamp:
          d.timestamp instanceof Date
            ? d.timestamp.toISOString()
            : (d.timestamp ?? now()),
      };
      this.builder.addDialogue(dialogue);
    }

    // Process all complete windows
    const units = await this.builder.processWindows();
    await this.saveAndIndexUnits(units);
  }

  /**
   * Finalize dialogue input, process any remaining buffer
   */
  async finalize(): Promise<void> {
    const units = await this.builder.processRemaining();
    await this.saveAndIndexUnits(units);
    this.index.rebuildLexicalIndex();
  }

  /**
   * Ask a question - Core Q&A interface
   *
   * @param question User question
   * @returns Answer based on memory
   */
  async ask(question: string): Promise<string> {
    await this.initialize();

    // Stage 3: Adaptive retrieval
    const context = await this.retriever.retrieve(question);
    this.events.emit("retrieval:context_retrieved", {
      query: question,
      unitCount: context.units.length,
      abstractCount: context.abstracts.length,
    });

    // Generate answer
    const answer = await this.generator.generate(question, context);
    this.events.emit("retrieval:answer_generated", {
      query: question,
      answer,
    });
    return answer;
  }

  /**
   * Search memories without generating an answer
   *
   * @param query Search query
   * @param options Search options
   * @returns Matching memory units
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryUnit[]> {
    await this.initialize();

    const results = await this.index.hybridSearch(
      query,
      options?.filter,
      options?.limit,
    );

    return results.map((r) => {
      const unit = { ...r.unit };
      if (!options?.includeEmbeddings) {
        delete unit.embedding;
      }
      return unit;
    });
  }

  /**
   * Retrieve context for a query (for custom answer generation)
   */
  async getContext(query: string): Promise<RetrievalContext> {
    await this.initialize();
    return this.retriever.retrieve(query);
  }

  /**
   * Add raw text content (bypasses dialogue windowing).
   * Extracts memory units via LLM from the text directly.
   *
   * @param text Text content to process
   * @param metadata Optional metadata (source, timestamp, topic)
   * @returns Created memory units
   */
  async addText(
    text: string,
    metadata?: TextMetadata,
  ): Promise<MemoryUnit[]> {
    const units = await this.builder.processText(
      text,
      metadata?.timestamp ?? now(),
    );
    await this.saveAndIndexUnits(units);
    return units;
  }

  /**
   * Add a long document by chunking and extracting memories from each chunk.
   *
   * @param content Document text content
   * @param options Chunking and metadata options
   * @returns All created memory units across chunks
   */
  async addDocument(
    content: string,
    options?: DocumentOptions,
  ): Promise<MemoryUnit[]> {
    const chunks = chunkText(
      content,
      options?.chunkSize ?? 2000,
      options?.chunkOverlap ?? 200,
    );

    const allUnits: MemoryUnit[] = [];
    for (const chunk of chunks) {
      const units = await this.addText(chunk, {
        source: options?.source,
        timestamp: options?.timestamp,
      });
      allUnits.push(...units);
    }
    return allUnits;
  }

  /**
   * Add a direct fact without LLM extraction.
   * Creates a memory unit directly from the provided content and metadata.
   *
   * @param content The fact to store (e.g., "User prefers dark mode")
   * @param metadata Optional metadata (persons, entities, topic, etc.)
   * @returns The created memory unit
   */
  async addFact(
    content: string,
    metadata?: FactMetadata,
  ): Promise<MemoryUnit> {
    const [embedding] = await this.embeddings.embed([content]);

    const unit: MemoryUnit = {
      id: uuidv4(),
      content,
      keywords: metadata?.keywords ?? content.toLowerCase().split(/\s+/).filter((w) => w.length > 2),
      timestamp: metadata?.timestamp ?? now(),
      location: metadata?.location,
      persons: metadata?.persons ?? [],
      entities: metadata?.entities ?? [],
      topic: metadata?.topic,
      salience: metadata?.salience ?? "medium",
      embedding,
      sourceDialogueIds: [],
      createdAt: now(),
    };

    await this.saveAndIndexUnits([unit]);
    return unit;
  }

  /**
   * Get all memory units
   */
  async getAllMemories(): Promise<MemoryUnit[]> {
    return this.storage.getAllUnits();
  }

  /**
   * Get memory count
   */
  async getMemoryCount(): Promise<number> {
    const units = await this.storage.getAllUnits();
    return units.length;
  }

  /**
   * Get system statistics
   */
  getStats(): {
    indexedUnits: number;
    bufferedDialogues: number;
    processedDialogues: number;
  } {
    const builderStats = this.builder.getStats();
    return {
      indexedUnits: this.index.size,
      bufferedDialogues: builderStats.bufferedDialogues,
      processedDialogues: builderStats.processedDialogues,
    };
  }

  /**
   * Export all memory data
   */
  async export(): Promise<ExportData> {
    return this.storage.export();
  }

  /**
   * Import memory data
   */
  async import(data: ExportData): Promise<void> {
    await this.storage.import(data);
    await this.index.addUnits(data.units);
    this.index.rebuildLexicalIndex();
    this.initialized = true;
  }

  /**
   * Manually trigger abstract memory consolidation.
   * Clusters similar memory units and generates abstract patterns.
   */
  async consolidate(): Promise<AbstractMemory[]> {
    await this.initialize();
    return this.abstractionEngine.consolidate();
  }

  /**
   * Clear all memories
   */
  async clear(): Promise<void> {
    await this.storage.clear();
    this.index.clear();
    this.builder.reset();
    this.abstractionEngine.reset();
    this.dialogueCounter = 0;
    this.initialized = false;
    this.events.emit("storage:cleared", {});
  }

  /**
   * Save units to storage and add to index
   */
  private async saveAndIndexUnits(units: MemoryUnit[]): Promise<void> {
    if (units.length === 0) return;

    await this.storage.saveUnits(units);
    await this.index.addUnits(units);

    this.events.emit("memory:units_created", { units, count: units.length });
    this.events.emit("memory:units_indexed", {
      unitIds: units.map((u) => u.id),
      count: units.length,
    });

    // Trigger abstract memory consolidation if interval reached
    const abstracts = await this.abstractionEngine.maybeConsolidate(
      units.length,
    );
    for (const abstract of abstracts) {
      this.events.emit("memory:abstract_created", {
        abstract,
        sourceCount: abstract.sourceUnitIds.length,
      });
    }
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a SimpleMem instance with minimal configuration
 */
export function createSimpleMem(options: SimpleMemOptions): SimpleMem {
  return new SimpleMem(options);
}
