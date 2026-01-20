/**
 * Stage 2: Structured Indexing
 *
 * Paper Reference: Section 3.2 - Structured Indexing
 *
 * Multi-view indexing across three layers:
 * 1. Semantic Layer: Dense vector embeddings for fuzzy matching
 * 2. Lexical Layer: Sparse representation for exact keyword/entity matches
 * 3. Symbolic Layer: Structured metadata for deterministic filtering
 */

import type {
  MemoryUnit,
  EmbeddingProvider,
  QueryFilter,
  SearchResult,
} from "../types/index.js";
import {
  cosineSimilarity,
  BM25Scorer,
  computeHybridScore,
} from "../utils/similarity.js";
import { isWithinRange } from "../utils/temporal.js";

// =============================================================================
// Configuration
// =============================================================================

export interface IndexingConfig {
  /**
   * Max results from semantic search
   * @default 25
   */
  semanticTopK: number;

  /**
   * Max results from keyword search
   * @default 5
   */
  keywordTopK: number;

  /**
   * Max results from structured search
   * @default 5
   */
  structuredTopK: number;

  /**
   * Weight for semantic similarity in hybrid scoring (α)
   * @default 0.6
   */
  semanticWeight: number;

  /**
   * Weight for lexical relevance in hybrid scoring (β)
   * @default 0.3
   */
  lexicalWeight: number;

  /**
   * Symbolic constraint bonus (γ)
   * @default 0.1
   */
  symbolicWeight: number;
}

export const DEFAULT_INDEXING_CONFIG: IndexingConfig = {
  semanticTopK: 25,
  keywordTopK: 5,
  structuredTopK: 5,
  semanticWeight: 0.6,
  lexicalWeight: 0.3,
  symbolicWeight: 0.1,
};

// =============================================================================
// Hybrid Index Class
// =============================================================================

/**
 * Hybrid Index - Multi-view indexing for memory retrieval
 *
 * Implements M(m_k) = {v_k (semantic), h_k (lexical), R_k (symbolic)}
 */
export class HybridIndex {
  private units: Map<string, MemoryUnit> = new Map();
  private embeddings: EmbeddingProvider;
  private bm25: BM25Scorer;
  private config: IndexingConfig;
  private needsRebuild = false;

  constructor(
    embeddings: EmbeddingProvider,
    config: Partial<IndexingConfig> = {},
  ) {
    this.embeddings = embeddings;
    this.config = { ...DEFAULT_INDEXING_CONFIG, ...config };
    this.bm25 = new BM25Scorer();
  }

  /**
   * Add memory units to the index
   */
  async addUnits(units: MemoryUnit[]): Promise<void> {
    // Generate embeddings for units that don't have them
    const unitsNeedingEmbeddings = units.filter((u) => !u.embedding);

    if (unitsNeedingEmbeddings.length > 0) {
      const texts = unitsNeedingEmbeddings.map((u) => u.content);
      const embeddings = await this.embeddings.embed(texts);

      for (let i = 0; i < unitsNeedingEmbeddings.length; i++) {
        unitsNeedingEmbeddings[i].embedding = embeddings[i];
      }
    }

    // Add to index
    for (const unit of units) {
      this.units.set(unit.id, unit);
    }

    this.needsRebuild = true;
  }

  /**
   * Remove a unit from the index
   */
  removeUnit(id: string): void {
    this.units.delete(id);
    this.needsRebuild = true;
  }

  /**
   * Rebuild the BM25 index (call after adding units)
   */
  rebuildLexicalIndex(): void {
    const documents = Array.from(this.units.values()).map((u) => u.content);
    this.bm25.addDocuments(documents);
    this.needsRebuild = false;
  }

  /**
   * Semantic search using embedding similarity
   *
   * Paper Reference: Section 3.3 - λ₁ · cos(e_q, v_k)
   */
  async semanticSearch(query: string, topK?: number): Promise<SearchResult[]> {
    const k = topK ?? this.config.semanticTopK;

    // Get query embedding
    const [queryEmbedding] = await this.embeddings.embed([query]);

    // Score all units
    const results: SearchResult[] = [];
    for (const unit of this.units.values()) {
      if (!unit.embedding) continue;

      const score = cosineSimilarity(queryEmbedding, unit.embedding);
      results.push({
        unit,
        score,
        matchType: "semantic",
      });
    }

    // Sort by score and take top-k
    return results.sort((a, b) => b.score - a.score).slice(0, k);
  }

  /**
   * Lexical search using BM25
   *
   * Paper Reference: Section 3.3 - λ₂ · BM25(q_lex, S_k)
   */
  keywordSearch(query: string, topK?: number): SearchResult[] {
    const k = topK ?? this.config.keywordTopK;

    if (this.needsRebuild) {
      this.rebuildLexicalIndex();
    }

    const unitsArray = Array.from(this.units.values());
    const topResults = this.bm25.topK(query, k);

    return topResults.map(([index, score]) => ({
      unit: unitsArray[index],
      score,
      matchType: "lexical" as const,
    }));
  }

  /**
   * Structured search using metadata filters
   *
   * Paper Reference: Section 3.3 - γ · 𝕀(R_k ⊨ C_meta)
   */
  structuredSearch(filter: QueryFilter, topK?: number): SearchResult[] {
    const k = topK ?? this.config.structuredTopK;
    const results: SearchResult[] = [];

    for (const unit of this.units.values()) {
      if (this.matchesFilter(unit, filter)) {
        results.push({
          unit,
          score: 1.0, // Binary match
          matchType: "symbolic",
        });
      }
    }

    return results.slice(0, k);
  }

  /**
   * Hybrid search combining all three layers
   *
   * Paper Reference: Section 3.3
   * S(q, m_k) = α * cos(E(q), v_k) + β * BM25(q, m_k) + γ * 𝕀(constraints)
   */
  async hybridSearch(
    query: string,
    filter?: QueryFilter,
    topK?: number,
  ): Promise<SearchResult[]> {
    const k = topK ?? this.config.semanticTopK;

    // Get results from all layers
    const [semanticResults, keywordResults] = await Promise.all([
      this.semanticSearch(query, k * 2),
      Promise.resolve(this.keywordSearch(query, k * 2)),
    ]);

    // Normalize BM25 scores to [0, 1]
    const maxBM25 = Math.max(...keywordResults.map((r) => r.score), 1);

    // Create score maps
    const semanticScores = new Map<string, number>();
    const lexicalScores = new Map<string, number>();

    for (const r of semanticResults) {
      semanticScores.set(r.unit.id, r.score);
    }
    for (const r of keywordResults) {
      lexicalScores.set(r.unit.id, r.score / maxBM25);
    }

    // Compute hybrid scores for all unique units
    const allUnitIds = new Set([
      ...semanticScores.keys(),
      ...lexicalScores.keys(),
    ]);

    const hybridResults: SearchResult[] = [];

    for (const unitId of allUnitIds) {
      const unit = this.units.get(unitId);
      if (!unit) continue;

      const semanticScore = semanticScores.get(unitId) ?? 0;
      const lexicalScore = lexicalScores.get(unitId) ?? 0;
      const matchesConstraints = filter
        ? this.matchesFilter(unit, filter)
        : true;

      const hybridScore = computeHybridScore(
        semanticScore,
        lexicalScore,
        matchesConstraints,
        this.config.semanticWeight,
        this.config.lexicalWeight,
        this.config.symbolicWeight,
      );

      hybridResults.push({
        unit,
        score: hybridScore,
        matchType: "hybrid",
      });
    }

    // Sort and return top-k
    return hybridResults.sort((a, b) => b.score - a.score).slice(0, k);
  }

  /**
   * Get all indexed units
   */
  getAllUnits(): MemoryUnit[] {
    return Array.from(this.units.values());
  }

  /**
   * Get unit count
   */
  get size(): number {
    return this.units.size;
  }

  /**
   * Clear the index
   */
  clear(): void {
    this.units.clear();
    this.bm25 = new BM25Scorer();
    this.needsRebuild = false;
  }

  /**
   * Check if a unit matches filter criteria
   */
  private matchesFilter(unit: MemoryUnit, filter: QueryFilter): boolean {
    if (filter.persons && filter.persons.length > 0) {
      const hasMatch = filter.persons.some((p) =>
        unit.persons.some((up) => up.toLowerCase().includes(p.toLowerCase())),
      );
      if (!hasMatch) return false;
    }

    if (filter.entities && filter.entities.length > 0) {
      const hasMatch = filter.entities.some((e) =>
        unit.entities.some((ue) => ue.toLowerCase().includes(e.toLowerCase())),
      );
      if (!hasMatch) return false;
    }

    if (filter.timestampRange && unit.timestamp) {
      const { start, end } = filter.timestampRange;
      if (start && end && !isWithinRange(unit.timestamp, start, end)) {
        return false;
      }
    }

    if (filter.location && unit.location) {
      if (
        !unit.location.toLowerCase().includes(filter.location.toLowerCase())
      ) {
        return false;
      }
    }

    if (filter.topic && unit.topic) {
      if (!unit.topic.toLowerCase().includes(filter.topic.toLowerCase())) {
        return false;
      }
    }

    return true;
  }
}
