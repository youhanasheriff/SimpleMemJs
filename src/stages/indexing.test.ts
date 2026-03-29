import { describe, it, expect, beforeEach } from "vitest";
import { v4 as uuidv4 } from "uuid";
import type { EmbeddingProvider, MemoryUnit } from "../types/index";
import { HybridIndex } from "./indexing";

// =============================================================================
// Mock Provider
// =============================================================================

class MockEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions = 8;

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((text) => {
      const vec = new Array(this.dimensions).fill(0);
      for (let i = 0; i < text.length; i++) {
        vec[i % this.dimensions] += text.charCodeAt(i) / 1000;
      }
      const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
      return vec.map((v) => v / norm);
    });
  }
}

function makeUnit(overrides: Partial<MemoryUnit> = {}): MemoryUnit {
  return {
    id: uuidv4(),
    content: "Alice meets Bob at Starbucks on Monday",
    keywords: ["alice", "bob", "starbucks", "monday"],
    persons: ["Alice", "Bob"],
    entities: ["Starbucks"],
    sourceDialogueIds: [1, 2],
    salience: "medium",
    timestamp: "2025-06-15T14:00:00.000Z",
    location: "Starbucks Downtown",
    topic: "meeting",
    createdAt: "2025-06-15T12:00:00.000Z",
    ...overrides,
  };
}

// =============================================================================
// Tests
// =============================================================================

describe("HybridIndex", () => {
  let embeddings: MockEmbeddingProvider;
  let index: HybridIndex;

  beforeEach(() => {
    embeddings = new MockEmbeddingProvider();
    index = new HybridIndex(embeddings);
  });

  it("adds units and generates embeddings", async () => {
    const unit = makeUnit({ embedding: undefined });
    await index.addUnits([unit]);

    expect(index.size).toBe(1);
    const all = index.getAllUnits();
    expect(all[0].embedding).toBeDefined();
    expect(all[0].embedding!.length).toBe(embeddings.dimensions);
  });

  it("preserves existing embeddings", async () => {
    const existingEmbed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    const unit = makeUnit({ embedding: existingEmbed });
    await index.addUnits([unit]);

    const all = index.getAllUnits();
    expect(all[0].embedding).toEqual(existingEmbed);
  });

  it("semantic search returns scored results", async () => {
    await index.addUnits([
      makeUnit({ content: "Alice loves coffee at Starbucks" }),
      makeUnit({ content: "Bob plays tennis every Sunday" }),
      makeUnit({ content: "Alice orders cappuccino at the cafe" }),
    ]);

    const results = await index.semanticSearch("coffee at Starbucks", 2);
    expect(results).toHaveLength(2);
    expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
    expect(results[0].matchType).toBe("semantic");
  });

  it("keyword search returns BM25 results", async () => {
    await index.addUnits([
      makeUnit({ content: "Alice meets Bob at Starbucks" }),
      makeUnit({ content: "Charlie runs in the park" }),
    ]);
    index.rebuildLexicalIndex();

    const results = index.keywordSearch("Alice Starbucks");
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].matchType).toBe("lexical");
  });

  it("structured search filters by metadata", async () => {
    await index.addUnits([
      makeUnit({ persons: ["Alice"], location: "Starbucks" }),
      makeUnit({ persons: ["Bob"], location: "Office" }),
    ]);

    const results = index.structuredSearch({ persons: ["Alice"] });
    expect(results).toHaveLength(1);
    expect(results[0].unit.persons).toContain("Alice");
    expect(results[0].matchType).toBe("symbolic");
  });

  it("hybrid search combines all layers", async () => {
    await index.addUnits([
      makeUnit({
        content: "Alice meets Bob at Starbucks for coffee",
        persons: ["Alice", "Bob"],
      }),
      makeUnit({
        content: "Charlie runs in the park every morning",
        persons: ["Charlie"],
      }),
    ]);
    index.rebuildLexicalIndex();

    const results = await index.hybridSearch("Alice Starbucks coffee", {
      persons: ["Alice"],
    });
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].matchType).toBe("hybrid");
  });

  it("removeUnit works correctly", async () => {
    const unit = makeUnit();
    await index.addUnits([unit]);
    expect(index.size).toBe(1);

    index.removeUnit(unit.id);
    expect(index.size).toBe(0);
  });

  it("clear empties the index", async () => {
    await index.addUnits([makeUnit(), makeUnit()]);
    expect(index.size).toBe(2);

    index.clear();
    expect(index.size).toBe(0);
  });
});
