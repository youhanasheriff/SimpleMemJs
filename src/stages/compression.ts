/**
 * Stage 1: Semantic Structured Compression
 *
 * Paper Reference: Section 3.1 - Semantic Structured Compression
 *
 * Transforms raw dialogue into atomic, self-contained memory units:
 * 1. Sliding window segmentation with entropy-aware filtering
 * 2. Context normalization (coreference resolution, temporal anchoring)
 * 3. Memory unit extraction via LLM
 */

import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type {
  Dialogue,
  MemoryUnit,
  LLMProvider,
  EmbeddingProvider,
} from "../types/index.js";
import { cosineSimilarity } from "../utils/similarity.js";
import { now } from "../utils/temporal.js";

// =============================================================================
// Configuration
// =============================================================================

export interface CompressionConfig {
  /**
   * Number of dialogues per sliding window
   * @default 40
   */
  windowSize: number;

  /**
   * Window overlap for context continuity
   * @default 2
   */
  overlapSize: number;

  /**
   * Threshold for filtering low-information windows (τ_redundant)
   * Windows with information score below this are discarded
   * @default 0.3
   */
  redundancyThreshold: number;

  /**
   * Weight for entity novelty vs semantic divergence (α)
   * @default 0.5
   */
  entityWeight: number;
}

export const DEFAULT_COMPRESSION_CONFIG: CompressionConfig = {
  windowSize: 40,
  overlapSize: 2,
  redundancyThreshold: 0.3,
  entityWeight: 0.5,
};

// =============================================================================
// LLM Response Schema
// =============================================================================

const ExtractedMemorySchema = z.object({
  content: z.string(),
  keywords: z.array(z.string()).default([]),
  timestamp: z.string().nullable().optional(),
  location: z.string().nullable().optional(),
  persons: z.array(z.string()).default([]),
  entities: z.array(z.string()).default([]),
  topic: z.string().nullable().optional(),
  salience: z.enum(["high", "medium", "low"]).default("medium"),
});

const ExtractionResponseSchema = z.object({
  memory_units: z.array(ExtractedMemorySchema),
});

// =============================================================================
// Memory Builder Class
// =============================================================================

/**
 * Memory Builder - Stage 1: Semantic Structured Compression
 *
 * Paper Reference: Section 3.1 - Semantic Structured Compression
 *
 * Core Functions:
 * 1. Entropy-based filtering (implicit via window processing)
 * 2. Coreference resolution (Φ_coref)
 * 3. Temporal anchoring (Φ_time)
 * 4. Atomic fact extraction (Φ_extract)
 */
export class MemoryBuilder {
  private llm: LLMProvider;
  private embeddings: EmbeddingProvider;
  private config: CompressionConfig;
  private dialogueBuffer: Dialogue[] = [];
  private processedDialogueIds: Set<number> = new Set();
  private previousWindowEmbedding: number[] | null = null;
  private previousEntities: Set<string> = new Set();
  private windowCounter = 0;

  constructor(
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    config: Partial<CompressionConfig> = {},
  ) {
    this.llm = llm;
    this.embeddings = embeddings;
    this.config = { ...DEFAULT_COMPRESSION_CONFIG, ...config };
  }

  /**
   * Add a single dialogue to the buffer
   */
  addDialogue(dialogue: Dialogue): void {
    if (!this.processedDialogueIds.has(dialogue.id)) {
      this.dialogueBuffer.push(dialogue);
    }
  }

  /**
   * Add multiple dialogues to the buffer
   */
  addDialogues(dialogues: Dialogue[]): void {
    for (const dialogue of dialogues) {
      this.addDialogue(dialogue);
    }
  }

  /**
   * Check if buffer has enough dialogues for a full window
   */
  hasFullWindow(): boolean {
    return this.dialogueBuffer.length >= this.config.windowSize;
  }

  /**
   * Process all pending windows and return extracted memory units
   */
  async processWindows(): Promise<MemoryUnit[]> {
    const allUnits: MemoryUnit[] = [];

    while (this.hasFullWindow()) {
      const window = this.extractWindow();
      const units = await this.processWindow(window);
      allUnits.push(...units);
    }

    return allUnits;
  }

  /**
   * Process remaining dialogues (less than a full window)
   */
  async processRemaining(): Promise<MemoryUnit[]> {
    if (this.dialogueBuffer.length === 0) {
      return [];
    }

    const window = [...this.dialogueBuffer];
    this.dialogueBuffer = [];

    for (const d of window) {
      this.processedDialogueIds.add(d.id);
    }

    return this.processWindow(window);
  }

  /**
   * Extract a window from the buffer (with overlap handling)
   */
  private extractWindow(): Dialogue[] {
    const window = this.dialogueBuffer.slice(0, this.config.windowSize);

    // Move buffer forward, keeping overlap
    const stride = this.config.windowSize - this.config.overlapSize;
    const consumed = this.dialogueBuffer.slice(0, stride);
    this.dialogueBuffer = this.dialogueBuffer.slice(stride);

    // Mark consumed dialogues as processed
    for (const d of consumed) {
      this.processedDialogueIds.add(d.id);
    }

    return window;
  }

  /**
   * Process a single window and extract memory units
   *
   * De-linearization Transformation F_θ: W_t → {m_k}
   * F_θ = Φ_time ∘ Φ_coref ∘ Φ_extract
   */
  private async processWindow(window: Dialogue[]): Promise<MemoryUnit[]> {
    this.windowCounter++;

    // Step 1: Compute information score for entropy-aware filtering
    const infoScore = await this.computeInformationScore(window);

    if (infoScore < this.config.redundancyThreshold) {
      // Low-information window - skip processing
      return [];
    }

    // Step 2: Build context from previous window
    const context = this.buildPreviousContext();

    // Step 3: Extract memory units via LLM
    const units = await this.extractMemoryUnits(window, context);

    // Step 4: Update state for next window
    await this.updateWindowState(window);

    return units;
  }

  /**
   * Compute information score for entropy-aware filtering
   *
   * Paper Reference: Section 3.1 - Information Filtering
   * H(W_t) = α · |ℰ_new| + (1-α) · (1 - cos(E(W_t), E(H_prev)))
   */
  private async computeInformationScore(window: Dialogue[]): Promise<number> {
    // Extract entities from current window
    const windowText = window
      .map((d) => `${d.speaker}: ${d.content}`)
      .join("\n");
    const currentEntities = this.extractBasicEntities(windowText);

    // Count new entities not seen before
    const newEntities = currentEntities.filter(
      (e) => !this.previousEntities.has(e.toLowerCase()),
    );
    const entityScore = Math.min(newEntities.length / 5, 1); // Normalize to [0,1]

    // Compute semantic divergence from previous window
    let semanticScore = 1.0; // Default: maximum novelty

    if (this.previousWindowEmbedding) {
      const [currentEmbedding] = await this.embeddings.embed([windowText]);
      const similarity = cosineSimilarity(
        currentEmbedding,
        this.previousWindowEmbedding,
      );
      semanticScore = 1 - similarity; // Divergence = 1 - similarity
    }

    // Combined score
    const alpha = this.config.entityWeight;
    return alpha * entityScore + (1 - alpha) * semanticScore;
  }

  /**
   * Simple entity extraction (capital words as a heuristic)
   */
  private extractBasicEntities(text: string): string[] {
    const words = text.split(/\s+/);
    const entities: string[] = [];

    for (const word of words) {
      const cleaned = word.replace(/[^\w]/g, "");
      if (
        cleaned.length > 1 &&
        cleaned[0] === cleaned[0].toUpperCase() &&
        cleaned[0] !== cleaned[0].toLowerCase()
      ) {
        entities.push(cleaned);
      }
    }

    return [...new Set(entities)];
  }

  /**
   * Build context string from previous extractions
   */
  private buildPreviousContext(): string {
    if (this.previousEntities.size === 0) {
      return "No previous context available.";
    }

    return `Previously mentioned entities: ${[...this.previousEntities].slice(0, 20).join(", ")}`;
  }

  /**
   * Update state after processing a window
   */
  private async updateWindowState(window: Dialogue[]): Promise<void> {
    const windowText = window
      .map((d) => `${d.speaker}: ${d.content}`)
      .join("\n");

    // Update embedding
    const [embedding] = await this.embeddings.embed([windowText]);
    this.previousWindowEmbedding = embedding;

    // Update entities
    const entities = this.extractBasicEntities(windowText);
    for (const e of entities) {
      this.previousEntities.add(e.toLowerCase());
    }
  }

  /**
   * Extract memory units from a window via LLM
   */
  private async extractMemoryUnits(
    window: Dialogue[],
    context: string,
  ): Promise<MemoryUnit[]> {
    const dialogueText = window
      .map((d) => {
        const timeStr = d.timestamp ? `[${d.timestamp}] ` : "";
        return `${timeStr}${d.speaker}: ${d.content}`;
      })
      .join("\n");

    const dialogueIds = window.map((d) => d.id);
    const windowStartTime = window[0]?.timestamp ?? now();

    const prompt = this.buildExtractionPrompt(
      dialogueText,
      windowStartTime,
      context,
    );

    try {
      const response = await this.llm.completeJSON(
        prompt,
        ExtractionResponseSchema,
      );

      return response.memory_units.map(
        (unit) =>
          ({
            id: uuidv4(),
            content: unit.content,
            keywords: unit.keywords ?? [], // Fix: Ensure array not undefined
            timestamp: unit.timestamp ?? undefined,
            location: unit.location ?? undefined,
            persons: unit.persons ?? [], // Fix: Ensure array not undefined
            entities: unit.entities ?? [], // Fix: Ensure array not undefined
            topic: unit.topic ?? undefined,
            salience: unit.salience,
            sourceDialogueIds: dialogueIds,
            createdAt: now(),
          }) as MemoryUnit,
      ); // Cast to satisfy strict check if needed, but values should align
    } catch (error) {
      console.error("Failed to extract memory units:", error);
      return [];
    }
  }

  /**
   * Build the LLM extraction prompt
   *
   * Based on Python SimpleMem prompt from memory_builder.py
   */
  private buildExtractionPrompt(
    dialogueText: string,
    windowStartTime: string,
    context: string,
  ): string {
    return `You are a memory encoder in a long-term memory system. Your task is to transform raw conversational input into compact, self-contained memory units.

INPUT METADATA:
Window Start Time: ${windowStartTime} (ISO 8601)

PREVIOUS CONTEXT:
${context}

CURRENT DIALOGUE:
${dialogueText}

INSTRUCTIONS:
1. Information Filtering:
   - Discard social filler, acknowledgements, and conversational routines that introduce no new factual or semantic information.
   - Discard redundant confirmations unless they modify or finalize a decision.
   - If no informative content is present, output an empty list.

2. Context Normalization (Coreference Resolution):
   - Resolve all pronouns and implicit references into explicit entity names.
   - Ensure each memory unit is interpretable without access to prior dialogue.
   - Example: "He said he'll do it" → "Bob said Bob will complete the report"

3. Temporal Normalization:
   - Convert relative temporal expressions (e.g., "tomorrow", "last week") into absolute ISO 8601 timestamps using the window start time.
   - Example: "tomorrow at 2pm" with start time 2025-11-15 → "2025-11-16T14:00:00"

4. Memory Unit Extraction:
   - Decompose complex utterances into minimal, indivisible factual statements.
   - Each memory unit should represent ONE atomic fact.

OUTPUT FORMAT (JSON):
{
  "memory_units": [
    {
      "content": "Alice agreed to meet Bob at the Starbucks on 5th Avenue on 2025-11-20T14:00:00.",
      "keywords": ["Alice", "Bob", "Starbucks", "meeting"],
      "timestamp": "2025-11-20T14:00:00",
      "location": "Starbucks, 5th Avenue",
      "persons": ["Alice", "Bob"],
      "entities": ["Starbucks"],
      "topic": "Meeting Planning",
      "salience": "high"
    }
  ]
}

Return ONLY the JSON object, no other text.`;
  }

  /**
   * Get statistics about the current state
   */
  getStats(): {
    bufferedDialogues: number;
    processedDialogues: number;
    windowsProcessed: number;
    knownEntities: number;
  } {
    return {
      bufferedDialogues: this.dialogueBuffer.length,
      processedDialogues: this.processedDialogueIds.size,
      windowsProcessed: this.windowCounter,
      knownEntities: this.previousEntities.size,
    };
  }

  /**
   * Reset the builder state
   */
  reset(): void {
    this.dialogueBuffer = [];
    this.processedDialogueIds.clear();
    this.previousWindowEmbedding = null;
    this.previousEntities.clear();
    this.windowCounter = 0;
  }
}
