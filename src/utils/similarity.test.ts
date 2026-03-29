import { describe, it, expect } from "vitest";
import {
  cosineSimilarity,
  euclideanDistance,
  BM25Scorer,
  computeAffinityScore,
  computeDynamicK,
  computeHybridScore,
} from "./similarity";

// =============================================================================
// cosineSimilarity
// =============================================================================

describe("cosineSimilarity", () => {
  it("returns 1 for identical vectors", () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1);
  });

  it("returns -1 for opposite vectors", () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  it("returns 0 for orthogonal vectors", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it("returns 0 when one vector is all zeros", () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0);
  });

  it("returns 0 when both vectors are all zeros", () => {
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
  });

  it("is insensitive to vector magnitude", () => {
    const a = cosineSimilarity([1, 2, 3], [4, 5, 6]);
    const b = cosineSimilarity([2, 4, 6], [4, 5, 6]);
    expect(a).toBeCloseTo(b);
  });

  it("throws on dimension mismatch", () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow(
      "Vector dimensions must match",
    );
  });

  it("handles single-element vectors", () => {
    expect(cosineSimilarity([5], [3])).toBeCloseTo(1);
    expect(cosineSimilarity([-5], [3])).toBeCloseTo(-1);
  });
});

// =============================================================================
// euclideanDistance
// =============================================================================

describe("euclideanDistance", () => {
  it("returns 0 for identical vectors", () => {
    expect(euclideanDistance([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  it("computes correct distance for simple cases", () => {
    expect(euclideanDistance([0, 0], [3, 4])).toBeCloseTo(5);
  });

  it("computes correct distance for 1D", () => {
    expect(euclideanDistance([0], [7])).toBeCloseTo(7);
  });

  it("is symmetric", () => {
    const d1 = euclideanDistance([1, 2, 3], [4, 5, 6]);
    const d2 = euclideanDistance([4, 5, 6], [1, 2, 3]);
    expect(d1).toBeCloseTo(d2);
  });

  it("throws on dimension mismatch", () => {
    expect(() => euclideanDistance([1], [1, 2])).toThrow(
      "Vector dimensions must match",
    );
  });

  it("handles zero vectors", () => {
    expect(euclideanDistance([0, 0], [0, 0])).toBe(0);
  });
});

// =============================================================================
// BM25Scorer
// =============================================================================

describe("BM25Scorer", () => {
  it("scores matching documents higher than non-matching", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments([
      "the cat sat on the mat",
      "the dog ran in the park",
      "a fish swam in the sea",
    ]);

    const scores = scorer.score("cat mat");
    expect(scores[0]).toBeGreaterThan(scores[1]);
    expect(scores[0]).toBeGreaterThan(scores[2]);
  });

  it("returns 0 for non-matching documents", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments(["hello world", "foo bar"]);

    const scores = scorer.score("xyz");
    expect(scores[0]).toBe(0);
    expect(scores[1]).toBe(0);
  });

  it("topK returns correct number of results", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments([
      "alpha beta gamma",
      "beta gamma delta",
      "gamma delta epsilon",
      "delta epsilon zeta",
    ]);

    const results = scorer.topK("beta", 2);
    expect(results).toHaveLength(2);
    expect(results[0][1]).toBeGreaterThanOrEqual(results[1][1]);
  });

  it("topK filters out zero-score documents", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments(["apple", "banana", "cherry"]);

    const results = scorer.topK("apple", 10);
    expect(results).toHaveLength(1);
    expect(results[0][0]).toBe(0); // index of "apple"
  });

  it("handles empty query", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments(["hello world"]);

    const scores = scorer.score("");
    expect(scores[0]).toBe(0);
  });

  it("handles punctuation in documents", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments(["hello, world!", "foo-bar"]);

    const scores = scorer.score("hello");
    expect(scores[0]).toBeGreaterThan(0);
  });

  it("is case-insensitive", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments(["Hello World"]);

    const scores = scorer.score("hello");
    expect(scores[0]).toBeGreaterThan(0);
  });

  it("respects custom k1 and b parameters", () => {
    const scorer1 = new BM25Scorer({ k1: 1.2, b: 0.75 });
    const scorer2 = new BM25Scorer({ k1: 2.0, b: 0.0 });

    const docs = ["the the the test", "test"];
    scorer1.addDocuments(docs);
    scorer2.addDocuments(docs);

    const scores1 = scorer1.score("test");
    const scores2 = scorer2.score("test");

    // Different params should yield different relative scores
    expect(scores1[0] / scores1[1]).not.toBeCloseTo(
      scores2[0] / scores2[1],
      1,
    );
  });

  it("handles single document", () => {
    const scorer = new BM25Scorer();
    scorer.addDocuments(["only document"]);

    const scores = scorer.score("only");
    expect(scores).toHaveLength(1);
    expect(scores[0]).toBeGreaterThan(0);
  });
});

// =============================================================================
// computeAffinityScore
// =============================================================================

describe("computeAffinityScore", () => {
  it("returns semanticSimilarity + beta when timeDiff is 0", () => {
    // exp(0) = 1, so score = similarity + beta * 1
    const score = computeAffinityScore(0.8, 0, 0.3, 86400);
    expect(score).toBeCloseTo(0.8 + 0.3);
  });

  it("temporal component decays with time", () => {
    const close = computeAffinityScore(0.5, 3600, 0.3, 86400); // 1 hour
    const far = computeAffinityScore(0.5, 864000, 0.3, 86400); // 10 days
    expect(close).toBeGreaterThan(far);
  });

  it("uses default beta=0.3 and gamma=86400", () => {
    const score = computeAffinityScore(0.5, 0);
    expect(score).toBeCloseTo(0.5 + 0.3);
  });

  it("handles negative time differences (uses absolute value)", () => {
    const pos = computeAffinityScore(0.5, 3600);
    const neg = computeAffinityScore(0.5, -3600);
    expect(pos).toBeCloseTo(neg);
  });

  it("temporal score approaches 0 for very large time differences", () => {
    const score = computeAffinityScore(0, 86400 * 365 * 10); // 10 years
    expect(score).toBeCloseTo(0, 1);
  });
});

// =============================================================================
// computeDynamicK
// =============================================================================

describe("computeDynamicK", () => {
  it("returns baseK when complexity is 0 (LOW)", () => {
    expect(computeDynamicK(3, 0)).toBe(3);
  });

  it("expands for HIGH complexity (complexity=1)", () => {
    // k_dyn = floor(3 * (1 + 2.0 * 1)) = floor(9) = 9
    expect(computeDynamicK(3, 1)).toBe(9);
  });

  it("uses default delta=2.0", () => {
    expect(computeDynamicK(3, 1)).toBe(9);
  });

  it("floors the result", () => {
    // k_dyn = floor(3 * (1 + 2.0 * 0.5)) = floor(6) = 6
    expect(computeDynamicK(3, 0.5)).toBe(6);
  });

  it("handles custom delta", () => {
    // k_dyn = floor(5 * (1 + 3.0 * 1)) = floor(20) = 20
    expect(computeDynamicK(5, 1, 3.0)).toBe(20);
  });
});

// =============================================================================
// computeHybridScore
// =============================================================================

describe("computeHybridScore", () => {
  it("computes weighted sum with default weights", () => {
    // S = 0.6*0.8 + 0.3*0.5 + 0.1 = 0.48 + 0.15 + 0.1 = 0.73
    const score = computeHybridScore(0.8, 0.5, true);
    expect(score).toBeCloseTo(0.73);
  });

  it("no symbolic bonus when constraints not matched", () => {
    // S = 0.6*0.8 + 0.3*0.5 + 0 = 0.63
    const score = computeHybridScore(0.8, 0.5, false);
    expect(score).toBeCloseTo(0.63);
  });

  it("uses custom weights", () => {
    // S = 0.5*1.0 + 0.4*0.0 + 0.1 = 0.6
    const score = computeHybridScore(1.0, 0.0, true, 0.5, 0.4, 0.1);
    expect(score).toBeCloseTo(0.6);
  });

  it("returns 0 when all inputs are 0 and no constraints", () => {
    expect(computeHybridScore(0, 0, false)).toBe(0);
  });

  it("returns only symbolic bonus when scores are 0", () => {
    const score = computeHybridScore(0, 0, true, 0.6, 0.3, 0.1);
    expect(score).toBeCloseTo(0.1);
  });
});
