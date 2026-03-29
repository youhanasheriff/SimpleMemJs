/**
 * Abstract Memory - Recursive Consolidation
 *
 * Paper Reference: Section 3.2 - Recursive Consolidation
 *
 * Clusters semantically similar memory units and generates
 * higher-level abstract patterns via LLM.
 */

import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type {
  MemoryUnit,
  AbstractMemory,
  LLMProvider,
  EmbeddingProvider,
  StorageAdapter,
  Logger,
} from "../types/index.js";
import { consoleLogger } from "../types/index.js";
import { cosineSimilarity, computeAffinityScore } from "../utils/similarity.js";
import { getTimeDiffSeconds, now } from "../utils/temporal.js";

// =============================================================================
// Configuration
// =============================================================================

export interface AbstractionConfig {
  /**
   * Affinity threshold for clustering (τ_cluster)
   * @default 0.75
   */
  clusterThreshold: number;

  /**
   * Temporal weight for affinity scoring (β)
   * @default 0.3
   */
  temporalBeta: number;

  /**
   * Temporal decay constant in seconds (γ)
   * @default 86400 (1 day)
   */
  temporalGamma: number;

  /**
   * Minimum units to form an abstract
   * @default 2
   */
  minClusterSize: number;

  /**
   * Maximum units in a cluster (prevents mega-clusters)
   * @default 20
   */
  maxClusterSize: number;

  /**
   * Number of new units before re-running consolidation
   * @default 50
   */
  consolidationInterval: number;
}

export const DEFAULT_ABSTRACTION_CONFIG: AbstractionConfig = {
  clusterThreshold: 0.75,
  temporalBeta: 0.3,
  temporalGamma: 86400,
  minClusterSize: 2,
  maxClusterSize: 20,
  consolidationInterval: 50,
};

// =============================================================================
// LLM Response Schema
// =============================================================================

const AbstractPatternSchema = z.object({
  pattern: z.string(),
  entities: z.array(z.string()).default([]),
  frequency: z.number().default(1),
});

// =============================================================================
// Abstraction Engine
// =============================================================================

export class AbstractionEngine {
  private llm: LLMProvider;
  private embeddings: EmbeddingProvider;
  private storage: StorageAdapter;
  private config: AbstractionConfig;
  private logger: Logger;
  private unitsSinceLastConsolidation = 0;

  constructor(
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    storage: StorageAdapter,
    config: Partial<AbstractionConfig> = {},
    logger: Logger = consoleLogger,
  ) {
    this.llm = llm;
    this.embeddings = embeddings;
    this.storage = storage;
    this.config = { ...DEFAULT_ABSTRACTION_CONFIG, ...config };
    this.logger = logger;
  }

  /**
   * Track new units and trigger consolidation when interval is reached.
   */
  async maybeConsolidate(newUnitCount: number): Promise<AbstractMemory[]> {
    this.unitsSinceLastConsolidation += newUnitCount;

    if (this.unitsSinceLastConsolidation < this.config.consolidationInterval) {
      return [];
    }

    this.unitsSinceLastConsolidation = 0;
    return this.consolidate();
  }

  /**
   * Run consolidation on all memory units.
   * Clusters similar units and generates abstract patterns via LLM.
   */
  async consolidate(): Promise<AbstractMemory[]> {
    const units = await this.storage.getAllUnits();
    if (units.length < this.config.minClusterSize) return [];

    // Only consider units with embeddings
    const embeddedUnits = units.filter((u) => u.embedding && u.embedding.length > 0);
    if (embeddedUnits.length < this.config.minClusterSize) return [];

    // Get existing abstracts to avoid re-clustering already-consolidated units
    const existingAbstracts = await this.storage.getAllAbstracts();
    const alreadyConsolidated = new Set(
      existingAbstracts.flatMap((a) => a.sourceUnitIds),
    );

    // Filter to only unconsolidated units
    const candidates = embeddedUnits.filter((u) => !alreadyConsolidated.has(u.id));
    if (candidates.length < this.config.minClusterSize) return [];

    // Step 1: Compute affinity matrix
    const affinityMatrix = this.computeAffinityMatrix(candidates);

    // Step 2: Agglomerative clustering
    const clusters = this.agglomerativeCluster(candidates, affinityMatrix);

    // Step 3: Generate abstracts from clusters
    const abstracts: AbstractMemory[] = [];
    for (const cluster of clusters) {
      if (cluster.length < this.config.minClusterSize) continue;

      try {
        const abstract = await this.abstractCluster(cluster);
        await this.storage.saveAbstract(abstract);
        abstracts.push(abstract);
      } catch (error) {
        this.logger.warn("Failed to abstract cluster", error);
      }
    }

    this.logger.info(
      `Consolidation complete: ${abstracts.length} abstracts from ${candidates.length} units`,
    );

    return abstracts;
  }

  /**
   * Compute pairwise affinity matrix for memory units.
   *
   * Paper: ω_ij = cos(v_i, v_j) + β · exp(−|t_i − t_j| / γ)
   */
  private computeAffinityMatrix(units: MemoryUnit[]): number[][] {
    const n = units.length;
    const matrix: number[][] = Array.from({ length: n }, () =>
      new Array(n).fill(0),
    );

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const semanticSim = cosineSimilarity(
          units[i].embedding!,
          units[j].embedding!,
        );

        const timeDiff =
          units[i].timestamp && units[j].timestamp
            ? getTimeDiffSeconds(units[i].timestamp!, units[j].timestamp!)
            : Infinity;

        const affinity = computeAffinityScore(
          semanticSim,
          timeDiff,
          this.config.temporalBeta,
          this.config.temporalGamma,
        );

        matrix[i][j] = affinity;
        matrix[j][i] = affinity;
      }
    }

    return matrix;
  }

  /**
   * Greedy agglomerative clustering with average linkage.
   * Merges clusters when inter-cluster affinity exceeds threshold.
   */
  private agglomerativeCluster(
    units: MemoryUnit[],
    affinityMatrix: number[][],
  ): MemoryUnit[][] {
    // Initialize: each unit is its own cluster
    let clusters: number[][] = units.map((_, i) => [i]);

    while (clusters.length > 1) {
      let bestAffinity = -Infinity;
      let bestI = -1;
      let bestJ = -1;

      // Find the two clusters with the highest average-linkage affinity
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          // Skip if merging would exceed max cluster size
          if (
            clusters[i].length + clusters[j].length >
            this.config.maxClusterSize
          ) {
            continue;
          }

          const avgAffinity = this.averageLinkage(
            clusters[i],
            clusters[j],
            affinityMatrix,
          );

          if (avgAffinity > bestAffinity) {
            bestAffinity = avgAffinity;
            bestI = i;
            bestJ = j;
          }
        }
      }

      // Stop if best affinity is below threshold
      if (bestAffinity < this.config.clusterThreshold) break;

      // Merge the two best clusters
      const merged = [...clusters[bestI], ...clusters[bestJ]];
      clusters = clusters.filter((_, idx) => idx !== bestI && idx !== bestJ);
      clusters.push(merged);
    }

    // Convert index clusters to unit clusters
    return clusters
      .filter((c) => c.length >= this.config.minClusterSize)
      .map((indices) => indices.map((i) => units[i]));
  }

  /**
   * Average linkage: mean affinity across all pairs between two clusters.
   */
  private averageLinkage(
    clusterA: number[],
    clusterB: number[],
    affinityMatrix: number[][],
  ): number {
    let totalAffinity = 0;
    let count = 0;

    for (const i of clusterA) {
      for (const j of clusterB) {
        totalAffinity += affinityMatrix[i][j];
        count++;
      }
    }

    return count > 0 ? totalAffinity / count : 0;
  }

  /**
   * Generate an abstract pattern from a cluster of memory units via LLM.
   */
  private async abstractCluster(
    units: MemoryUnit[],
  ): Promise<AbstractMemory> {
    const unitContents = units
      .map((u, i) => {
        const parts = [`[${i + 1}] ${u.content}`];
        if (u.timestamp) parts.push(`  Time: ${u.timestamp}`);
        if (u.location) parts.push(`  Location: ${u.location}`);
        return parts.join("\n");
      })
      .join("\n\n");

    const prompt = `You are a memory consolidation engine. Given a cluster of related memory units, extract a single higher-level pattern that captures the recurring theme or relationship.

MEMORY UNITS:
${unitContents}

Extract ONE abstract pattern. Output JSON:
{
  "pattern": "A concise description of the recurring pattern or relationship",
  "entities": ["entity1", "entity2"],
  "frequency": ${units.length}
}

Return ONLY the JSON.`;

    const response = await this.llm.completeJSON(prompt, AbstractPatternSchema);

    // Compute embedding for the abstract pattern
    const [embedding] = await this.embeddings.embed([response.pattern]);

    // Determine time range
    const timestamps = units
      .map((u) => u.timestamp)
      .filter((t): t is string => !!t)
      .sort();

    const responseEntities = response.entities ?? [];
    const allEntities = [
      ...new Set([
        ...responseEntities,
        ...units.flatMap((u) => u.entities),
      ]),
    ];

    return {
      id: uuidv4(),
      pattern: response.pattern,
      sourceUnitIds: units.map((u) => u.id),
      frequency: response.frequency ?? units.length,
      firstOccurrence: timestamps[0] ?? now(),
      lastOccurrence: timestamps[timestamps.length - 1] ?? now(),
      entities: allEntities,
      embedding,
      isArchived: false,
    };
  }

  /**
   * Reset the consolidation counter.
   */
  reset(): void {
    this.unitsSinceLastConsolidation = 0;
  }
}
