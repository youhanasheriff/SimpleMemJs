import { describe, it, expect, beforeEach } from "vitest";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type {
  LLMProvider,
  EmbeddingProvider,
  MemoryUnit,
  LLMCompletionOptions,
  RetrievalContext,
} from "../types/index";
import { HybridIndex } from "./indexing";
import { HybridRetriever, AnswerGenerator } from "./retrieval";

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
  private responses: Map<string, string> = new Map();
  private defaultResponse: string;

  constructor(defaultResponse = '{"reasoning":"ok","answer":"mock"}') {
    this.defaultResponse = defaultResponse;
  }

  setResponse(pattern: string, response: string): void {
    this.responses.set(pattern, response);
  }

  async complete(
    prompt: string,
    _options?: LLMCompletionOptions,
  ): Promise<string> {
    for (const [pattern, response] of this.responses) {
      if (prompt.includes(pattern)) return response;
    }
    return this.defaultResponse;
  }

  async completeJSON<T>(prompt: string, schema: z.ZodType<T>): Promise<T> {
    for (const [pattern, response] of this.responses) {
      if (prompt.includes(pattern)) return schema.parse(JSON.parse(response));
    }
    return schema.parse(JSON.parse(this.defaultResponse));
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
// HybridRetriever Tests
// =============================================================================

describe("HybridRetriever", () => {
  let embeddings: MockEmbeddingProvider;
  let index: HybridIndex;

  beforeEach(async () => {
    embeddings = new MockEmbeddingProvider();
    index = new HybridIndex(embeddings);

    await index.addUnits([
      makeUnit({
        content: "Alice meets Bob at Starbucks on 16 November 2025",
        persons: ["Alice", "Bob"],
        location: "Starbucks",
      }),
      makeUnit({
        content: "Charlie has a meeting at the office on Monday",
        persons: ["Charlie"],
        location: "Office",
      }),
    ]);
    index.rebuildLexicalIndex();
  });

  it("retrieves context with planning and reflection disabled", async () => {
    const llm = new MockLLMProvider();
    const retriever = new HybridRetriever(llm, index, {
      enablePlanning: false,
      enableReflection: false,
    });

    const context = await retriever.retrieve("Where does Alice meet Bob?");
    expect(context.units.length).toBeGreaterThanOrEqual(1);
    expect(context.totalTokens).toBeGreaterThan(0);
    expect(context.abstracts).toEqual([]);
  });

  it("analyzes query complexity", async () => {
    const analysisResponse = JSON.stringify({
      complexity: "LOW",
      retrieval_rationale: "Simple fact lookup",
      lexical_keywords: ["Alice", "Starbucks"],
      semantic_query: "Where does Alice meet Bob?",
    });
    const llm = new MockLLMProvider(analysisResponse);

    const retriever = new HybridRetriever(llm, index);
    const analysis = await retriever.analyzeQuery("Where does Alice meet Bob?");

    expect(analysis.complexity).toBe("LOW");
    expect(analysis.lexicalKeywords).toContain("Alice");
  });

  it("falls back gracefully when LLM analysis fails", async () => {
    const llm = new MockLLMProvider("not json");
    llm.completeJSON = async () => {
      throw new Error("LLM failure");
    };

    const retriever = new HybridRetriever(llm, index, {
      enableReflection: false,
    });
    const analysis = await retriever.analyzeQuery("test query");

    expect(analysis.complexity).toBe("LOW");
    expect(analysis.rationale).toBe("Default analysis");
  });

  it("uses dynamic K for HIGH complexity", async () => {
    const analysisResponse = JSON.stringify({
      complexity: "HIGH",
      retrieval_rationale: "Requires aggregation",
      lexical_keywords: ["meeting"],
      semantic_query: "meetings overview",
    });
    const llm = new MockLLMProvider(analysisResponse);

    const retriever = new HybridRetriever(llm, index, {
      baseK: 3,
      complexityDelta: 2.0,
      enableReflection: false,
    });

    const context = await retriever.retrieve("What meetings happened?");
    // We only have 2 units in the index
    expect(context.units.length).toBeLessThanOrEqual(2);
  });
});

// =============================================================================
// AnswerGenerator Tests
// =============================================================================

describe("AnswerGenerator", () => {
  it("returns no-info message when no units provided", async () => {
    const llm = new MockLLMProvider();
    const generator = new AnswerGenerator(llm);
    const context: RetrievalContext = {
      abstracts: [],
      units: [],
      totalTokens: 0,
    };

    const answer = await generator.generate("Where is Alice?", context);
    expect(answer).toContain("do not have enough information");
  });

  it("generates answer from context", async () => {
    const answerResponse = JSON.stringify({
      reasoning: "The context says Alice meets at Starbucks",
      answer: "Starbucks on 16 November 2025",
    });
    const llm = new MockLLMProvider(answerResponse);

    const generator = new AnswerGenerator(llm);
    const context: RetrievalContext = {
      abstracts: [],
      units: [
        makeUnit({
          content: "Alice meets Bob at Starbucks on 16 November 2025",
        }),
      ],
      totalTokens: 50,
    };

    const answer = await generator.generate(
      "Where does Alice meet Bob?",
      context,
    );
    expect(answer).toContain("Starbucks");
  });

  it("falls back to simple completion on JSON parse error", async () => {
    const llm = new MockLLMProvider("Starbucks on Monday");
    const origComplete = llm.complete.bind(llm);
    llm.completeJSON = async () => {
      throw new Error("JSON parse error");
    };
    llm.complete = origComplete;

    const generator = new AnswerGenerator(llm);
    const context: RetrievalContext = {
      abstracts: [],
      units: [makeUnit()],
      totalTokens: 50,
    };

    const answer = await generator.generate("Where?", context);
    expect(typeof answer).toBe("string");
    expect(answer.length).toBeGreaterThan(0);
  });
});
