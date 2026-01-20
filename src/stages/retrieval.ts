/**
 * Stage 3: Adaptive Query-Aware Retrieval
 *
 * Paper Reference: Section 3.3 - Adaptive Query-Aware Retrieval with Pruning
 *
 * Implements:
 * 1. Query complexity estimation (LOW/HIGH)
 * 2. Dynamic retrieval depth based on complexity
 * 3. Hybrid scoring across semantic, lexical, and symbolic layers
 * 4. Answer generation from retrieved context
 */

import { z } from "zod";
import type {
  MemoryUnit,
  LLMProvider,
  RetrievalContext,
  QueryFilter,
  QueryAnalysis,
} from "../types/index.js";
import { HybridIndex } from "./indexing.js";
import { computeDynamicK } from "../utils/similarity.js";

// =============================================================================
// Configuration
// =============================================================================

export interface RetrievalConfig {
  /**
   * Base number of results to retrieve (k_base)
   * @default 3
   */
  baseK: number;

  /**
   * Complexity expansion factor (δ)
   * @default 2.0
   */
  complexityDelta: number;

  /**
   * Whether to use query planning/analysis
   * @default true
   */
  enablePlanning: boolean;

  /**
   * Whether to use reflection (multi-round retrieval)
   * @default true
   */
  enableReflection: boolean;

  /**
   * Maximum reflection rounds
   * @default 2
   */
  maxReflectionRounds: number;
}

export const DEFAULT_RETRIEVAL_CONFIG: RetrievalConfig = {
  baseK: 3,
  complexityDelta: 2.0,
  enablePlanning: true,
  enableReflection: true,
  maxReflectionRounds: 2,
};

// =============================================================================
// Query Analysis Schema
// =============================================================================

const QueryAnalysisResponseSchema = z.object({
  complexity: z.enum(["LOW", "HIGH"]),
  retrieval_rationale: z.string(),
  lexical_keywords: z.array(z.string()),
  temporal_constraints: z
    .object({
      start: z.string().optional(),
      end: z.string().optional(),
    })
    .optional(),
  semantic_query: z.string(),
});

// =============================================================================
// Hybrid Retriever Class
// =============================================================================

/**
 * Hybrid Retriever - Stage 3: Adaptive Query-Aware Retrieval
 *
 * Implements complexity-aware retrieval with dynamic depth
 */
export class HybridRetriever {
  private llm: LLMProvider;
  private index: HybridIndex;
  private config: RetrievalConfig;

  constructor(
    llm: LLMProvider,
    index: HybridIndex,
    config: Partial<RetrievalConfig> = {},
  ) {
    this.llm = llm;
    this.index = index;
    this.config = { ...DEFAULT_RETRIEVAL_CONFIG, ...config };
  }

  /**
   * Retrieve relevant memory units for a query
   */
  async retrieve(query: string): Promise<RetrievalContext> {
    // Step 1: Analyze query if planning is enabled
    let analysis: QueryAnalysis | null = null;
    let filter: QueryFilter | undefined;

    if (this.config.enablePlanning) {
      analysis = await this.analyzeQuery(query);

      // Build filter from analysis
      if (
        analysis.temporalConstraints?.start ||
        analysis.temporalConstraints?.end
      ) {
        filter = {
          timestampRange: analysis.temporalConstraints,
        };
      }
    }

    // Step 2: Compute dynamic retrieval depth
    const complexityScore = analysis?.complexity === "HIGH" ? 1.0 : 0.0;
    const dynamicK = computeDynamicK(
      this.config.baseK,
      complexityScore,
      this.config.complexityDelta,
    );

    // Step 3: Execute hybrid search
    const searchQuery = analysis?.semanticQuery ?? query;
    const results = await this.index.hybridSearch(
      searchQuery,
      filter,
      dynamicK,
    );

    // Step 4: Optional reflection for more context
    let allUnits = results.map((r) => r.unit);

    if (this.config.enableReflection && analysis?.complexity === "HIGH") {
      allUnits = await this.reflectionSearch(query, allUnits);
    }

    // Step 5: Build retrieval context
    const totalTokens = this.estimateTokens(allUnits);

    return {
      abstracts: [], // TODO: Add abstract memories
      units: allUnits,
      totalTokens,
      retrievalRationale: analysis?.rationale,
    };
  }

  /**
   * Analyze query to determine complexity and extract signals
   *
   * Paper Reference: Section 3.3 - Query Complexity Estimation
   */
  async analyzeQuery(query: string): Promise<QueryAnalysis> {
    const prompt = `Analyze the following user query and generate a retrieval plan. Your objective is to retrieve sufficient information while minimizing unnecessary context usage.

USER QUERY:
${query}

INSTRUCTIONS:
1. Query Complexity Estimation:
   - Assign "LOW" if the query can be answered via direct fact lookup or a single memory unit.
   - Assign "HIGH" if the query requires aggregation across multiple events, temporal comparison, or synthesis of patterns.

2. Retrieval Signals:
   - Lexical layer: extract exact keywords or entity names.
   - Temporal layer: infer absolute time ranges if relevant (use ISO 8601 format).
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

Return ONLY the JSON object.`;

    try {
      const response = await this.llm.completeJSON(
        prompt,
        QueryAnalysisResponseSchema,
      );

      return {
        complexity: response.complexity,
        rationale: response.retrieval_rationale,
        lexicalKeywords: response.lexical_keywords,
        temporalConstraints: response.temporal_constraints,
        semanticQuery: response.semantic_query,
      };
    } catch (error) {
      // Fallback: assume LOW complexity
      return {
        complexity: "LOW",
        rationale: "Default analysis",
        lexicalKeywords: query.split(/\s+/).filter((w) => w.length > 3),
        semanticQuery: query,
      };
    }
  }

  /**
   * Reflection-based additional retrieval
   *
   * Performs additional searches if initial results are insufficient
   */
  private async reflectionSearch(
    originalQuery: string,
    initialResults: MemoryUnit[],
  ): Promise<MemoryUnit[]> {
    const allUnits = [...initialResults];
    const seenIds = new Set(initialResults.map((u) => u.id));

    for (let round = 0; round < this.config.maxReflectionRounds; round++) {
      // Check if we have enough context
      const isAdequate = await this.checkAdequacy(originalQuery, allUnits);
      if (isAdequate) break;

      // Generate additional queries
      const additionalQueries = await this.generateAdditionalQueries(
        originalQuery,
        allUnits,
      );

      if (additionalQueries.length === 0) break;

      // Execute additional searches
      for (const q of additionalQueries) {
        const results = await this.index.hybridSearch(q, undefined, 3);
        for (const r of results) {
          if (!seenIds.has(r.unit.id)) {
            allUnits.push(r.unit);
            seenIds.add(r.unit.id);
          }
        }
      }
    }

    return allUnits;
  }

  /**
   * Check if current context is adequate to answer the query
   */
  private async checkAdequacy(
    query: string,
    contexts: MemoryUnit[],
  ): Promise<boolean> {
    if (contexts.length === 0) return false;
    if (contexts.length >= 10) return true; // Enough context

    const contextStr = contexts
      .slice(0, 5)
      .map((c) => c.content)
      .join("\n");

    const prompt = `Given this query and the available context, determine if there is sufficient information to answer.

Query: ${query}

Available Context:
${contextStr}

Is this context sufficient? Reply with ONLY "yes" or "no".`;

    try {
      const response = await this.llm.complete(prompt, { temperature: 0.1 });
      return response.toLowerCase().includes("yes");
    } catch {
      return true; // Assume adequate on error
    }
  }

  /**
   * Generate additional search queries for reflection
   */
  private async generateAdditionalQueries(
    originalQuery: string,
    currentContexts: MemoryUnit[],
  ): Promise<string[]> {
    const contextSummary = currentContexts
      .slice(0, 3)
      .map((c) => c.content)
      .join("\n");

    const prompt = `Original query: ${originalQuery}

Current context available:
${contextSummary}

What additional information might be needed? Generate 1-2 follow-up search queries that could help answer the original query. Return as JSON array of strings.

Example: ["meeting time with Alice", "location of the discussion"]`;

    try {
      const response = await this.llm.complete(prompt, { temperature: 0.3 });

      // Parse JSON array
      const cleaned = response.trim();
      const parsed = JSON.parse(cleaned);

      if (Array.isArray(parsed)) {
        return parsed.filter((q: unknown) => typeof q === "string").slice(0, 2);
      }
    } catch {
      // Ignore errors
    }

    return [];
  }

  /**
   * Estimate token count for memory units
   */
  private estimateTokens(units: MemoryUnit[]): number {
    // Rough estimate: 1 token ≈ 4 characters
    const totalChars = units.reduce((sum, u) => {
      return (
        sum +
        u.content.length +
        (u.location?.length ?? 0) +
        u.persons.join(" ").length
      );
    }, 0);

    return Math.ceil(totalChars / 4);
  }
}

// =============================================================================
// Answer Generator
// =============================================================================

/**
 * Generate answers from retrieved context
 *
 * Paper Reference: Section 3.3 - Reconstructive Synthesis
 */
export class AnswerGenerator {
  private llm: LLMProvider;

  constructor(llm: LLMProvider) {
    this.llm = llm;
  }

  /**
   * Generate an answer from the retrieval context
   */
  async generate(query: string, context: RetrievalContext): Promise<string> {
    if (context.units.length === 0) {
      return "I do not have enough information in my memory to answer this question.";
    }

    const contextStr = this.formatContext(context);
    const prompt = this.buildPrompt(query, contextStr);

    const AnswerSchema = z.object({
      reasoning: z.string(),
      answer: z.string(),
    });

    try {
      const response = await this.llm.completeJSON(prompt, AnswerSchema);
      return response.answer;
    } catch (error) {
      // Fallback to simple completion
      const response = await this.llm.complete(prompt, { temperature: 0.1 });
      return response;
    }
  }

  /**
   * Format retrieval context for the prompt
   */
  private formatContext(context: RetrievalContext): string {
    const parts: string[] = [];

    for (let i = 0; i < context.units.length; i++) {
      const unit = context.units[i];
      const lines = [`[Context ${i + 1}]`, `Content: ${unit.content}`];

      if (unit.timestamp) lines.push(`Time: ${unit.timestamp}`);
      if (unit.location) lines.push(`Location: ${unit.location}`);
      if (unit.persons.length > 0)
        lines.push(`Persons: ${unit.persons.join(", ")}`);
      if (unit.entities.length > 0)
        lines.push(`Entities: ${unit.entities.join(", ")}`);
      if (unit.topic) lines.push(`Topic: ${unit.topic}`);

      parts.push(lines.join("\n"));
    }

    return parts.join("\n\n");
  }

  /**
   * Build the answer generation prompt
   */
  private buildPrompt(query: string, contextStr: string): string {
    return `Answer the user's question based on the provided context.

User Question: ${query}

Relevant Context:
${contextStr}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' when appropriate
5. Return your response in JSON format

Output Format:
{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
{
  "reasoning": "The context explicitly states the meeting time as 2025-11-16T14:00:00",
  "answer": "16 November 2025 at 2:00 PM"
}

Return ONLY the JSON, no other text.`;
  }
}
