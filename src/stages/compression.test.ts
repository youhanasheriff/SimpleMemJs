import { describe, it, expect, beforeEach } from "vitest";
import { z } from "zod";
import type {
  LLMProvider,
  EmbeddingProvider,
  LLMCompletionOptions,
} from "../types/index";
import { MemoryBuilder } from "./compression";

// =============================================================================
// Mock Providers
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

class MockLLMProvider implements LLMProvider {
  private defaultResponse: string;

  constructor(defaultResponse = "{}") {
    this.defaultResponse = defaultResponse;
  }

  async complete(
    _prompt: string,
    _options?: LLMCompletionOptions,
  ): Promise<string> {
    return this.defaultResponse;
  }

  async completeJSON<T>(_prompt: string, schema: z.ZodType<T>): Promise<T> {
    return schema.parse(JSON.parse(this.defaultResponse));
  }
}

// =============================================================================
// Tests
// =============================================================================

describe("MemoryBuilder", () => {
  let llm: MockLLMProvider;
  let embeddings: MockEmbeddingProvider;

  const VALID_RESPONSE = JSON.stringify({
    memory_units: [
      {
        content: "Alice will meet Bob at Starbucks",
        keywords: ["Alice", "Bob", "Starbucks"],
        timestamp: "2025-11-16T14:00:00",
        location: "Starbucks",
        persons: ["Alice", "Bob"],
        entities: ["Starbucks"],
        topic: "meeting",
        salience: "high",
      },
    ],
  });

  beforeEach(() => {
    llm = new MockLLMProvider(VALID_RESPONSE);
    embeddings = new MockEmbeddingProvider();
  });

  it("buffers dialogues until window is full", () => {
    const builder = new MemoryBuilder(llm, embeddings, { windowSize: 3 });
    builder.addDialogue({ id: 1, speaker: "Alice", content: "Hello" });
    builder.addDialogue({ id: 2, speaker: "Bob", content: "Hi" });
    expect(builder.hasFullWindow()).toBe(false);

    builder.addDialogue({ id: 3, speaker: "Alice", content: "How are you?" });
    expect(builder.hasFullWindow()).toBe(true);
  });

  it("skips already-processed dialogues", async () => {
    const builder = new MemoryBuilder(llm, embeddings, {
      windowSize: 2,
      overlapSize: 0,
      redundancyThreshold: 0,
    });

    builder.addDialogue({ id: 1, speaker: "A", content: "Hello" });
    builder.addDialogue({ id: 2, speaker: "B", content: "World" });
    await builder.processWindows();

    builder.addDialogue({ id: 1, speaker: "A", content: "Hello" });
    expect(builder.getStats().bufferedDialogues).toBe(0);
  });

  it("processWindows extracts memory units via LLM", async () => {
    const builder = new MemoryBuilder(llm, embeddings, {
      windowSize: 2,
      overlapSize: 0,
      redundancyThreshold: 0,
    });

    builder.addDialogue({
      id: 1,
      speaker: "Alice",
      content: "Let's meet tomorrow at Starbucks",
      timestamp: "2025-11-15T14:00:00",
    });
    builder.addDialogue({
      id: 2,
      speaker: "Bob",
      content: "Sure, see you there",
      timestamp: "2025-11-15T14:01:00",
    });

    const units = await builder.processWindows();
    expect(units.length).toBeGreaterThanOrEqual(1);
    expect(units[0].content).toContain("Alice");
    expect(units[0].persons).toContain("Alice");
  });

  it("processRemaining handles leftover buffer", async () => {
    const builder = new MemoryBuilder(llm, embeddings, {
      windowSize: 10,
      redundancyThreshold: 0,
    });

    builder.addDialogue({ id: 1, speaker: "Alice", content: "One message" });
    expect(builder.hasFullWindow()).toBe(false);

    const units = await builder.processRemaining();
    expect(units.length).toBeGreaterThanOrEqual(1);
  });

  it("returns empty array when LLM fails", async () => {
    const failingLlm = new MockLLMProvider("bad");
    failingLlm.completeJSON = async () => {
      throw new Error("LLM parse failure");
    };

    const builder = new MemoryBuilder(failingLlm, embeddings, {
      windowSize: 2,
      redundancyThreshold: 0,
    });
    builder.addDialogue({ id: 1, speaker: "A", content: "Hello" });
    builder.addDialogue({ id: 2, speaker: "B", content: "World" });

    const units = await builder.processWindows();
    expect(units).toEqual([]);
  });

  it("reset clears all state", () => {
    const builder = new MemoryBuilder(llm, embeddings);
    builder.addDialogue({ id: 1, speaker: "A", content: "Hello" });
    builder.reset();

    const stats = builder.getStats();
    expect(stats.bufferedDialogues).toBe(0);
    expect(stats.processedDialogues).toBe(0);
    expect(stats.windowsProcessed).toBe(0);
    expect(stats.knownEntities).toBe(0);
  });

  it("getStats returns accurate counts", () => {
    const builder = new MemoryBuilder(llm, embeddings, { windowSize: 5 });
    builder.addDialogue({ id: 1, speaker: "A", content: "Hello" });
    builder.addDialogue({ id: 2, speaker: "B", content: "World" });

    const stats = builder.getStats();
    expect(stats.bufferedDialogues).toBe(2);
    expect(stats.processedDialogues).toBe(0);
    expect(stats.windowsProcessed).toBe(0);
  });
});
