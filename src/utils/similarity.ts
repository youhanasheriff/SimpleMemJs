/**
 * Similarity computation utilities
 *
 * Implements cosine similarity and BM25 scoring for hybrid retrieval
 */

/**
 * Compute cosine similarity between two vectors
 *
 * @param a First vector
 * @param b Second vector
 * @returns Similarity score in range [-1, 1]
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimensions must match: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}

/**
 * Compute Euclidean distance between two vectors
 *
 * @param a First vector
 * @param b Second vector
 * @returns Distance (lower is more similar)
 */
export function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimensions must match: ${a.length} vs ${b.length}`);
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

/**
 * BM25 scoring parameters
 */
export interface BM25Params {
  k1?: number; // Term frequency saturation, default 1.2
  b?: number; // Length normalization, default 0.75
}

/**
 * Simple BM25 scorer for keyword matching
 *
 * Reference: Robertson & Zaragoza, 2009
 */
export class BM25Scorer {
  private k1: number;
  private b: number;
  private documents: string[][];
  private avgDocLength: number;
  private idf: Map<string, number>;

  constructor(params: BM25Params = {}) {
    this.k1 = params.k1 ?? 1.2;
    this.b = params.b ?? 0.75;
    this.documents = [];
    this.avgDocLength = 0;
    this.idf = new Map();
  }

  /**
   * Tokenize a text into terms (simple whitespace + lowercase)
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 0);
  }

  /**
   * Add documents to build the index
   */
  addDocuments(docs: string[]): void {
    this.documents = docs.map((d) => this.tokenize(d));

    // Calculate average document length
    const totalLength = this.documents.reduce(
      (sum, doc) => sum + doc.length,
      0,
    );
    this.avgDocLength = totalLength / this.documents.length || 1;

    // Calculate IDF for each term
    const docFreq = new Map<string, number>();
    for (const doc of this.documents) {
      const uniqueTerms = new Set(doc);
      for (const term of uniqueTerms) {
        docFreq.set(term, (docFreq.get(term) ?? 0) + 1);
      }
    }

    const N = this.documents.length;
    for (const [term, df] of docFreq) {
      // IDF with smoothing
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
      this.idf.set(term, idf);
    }
  }

  /**
   * Score a query against all documents
   *
   * @param query Query string
   * @returns Array of scores (same length as documents)
   */
  score(query: string): number[] {
    const queryTerms = this.tokenize(query);

    return this.documents.map((doc) => {
      let score = 0;
      const docLength = doc.length;

      // Count term frequencies in document
      const termFreq = new Map<string, number>();
      for (const term of doc) {
        termFreq.set(term, (termFreq.get(term) ?? 0) + 1);
      }

      for (const term of queryTerms) {
        const tf = termFreq.get(term) ?? 0;
        const idf = this.idf.get(term) ?? 0;

        if (tf > 0 && idf > 0) {
          // BM25 term score
          const numerator = tf * (this.k1 + 1);
          const denominator =
            tf +
            this.k1 * (1 - this.b + this.b * (docLength / this.avgDocLength));
          score += idf * (numerator / denominator);
        }
      }

      return score;
    });
  }

  /**
   * Get top-k documents by score
   *
   * @param query Query string
   * @param k Number of results
   * @returns Array of [index, score] pairs
   */
  topK(query: string, k: number): Array<[number, number]> {
    const scores = this.score(query);

    return scores
      .map((score, index): [number, number] => [index, score])
      .filter(([, score]) => score > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }
}

/**
 * Compute affinity score between two memory units
 *
 * Paper Reference: Section 3.2
 * ω_ij = cos(v_i, v_j) + β * exp(-|t_i - t_j| / γ)
 *
 * @param semanticSimilarity Cosine similarity of embeddings
 * @param timeDiffSeconds Time difference in seconds
 * @param beta Temporal weight (default: 0.3)
 * @param gamma Temporal decay constant in seconds (default: 86400 = 1 day)
 */
export function computeAffinityScore(
  semanticSimilarity: number,
  timeDiffSeconds: number,
  beta: number = 0.3,
  gamma: number = 86400,
): number {
  const temporalScore = Math.exp(-Math.abs(timeDiffSeconds) / gamma);
  return semanticSimilarity + beta * temporalScore;
}

/**
 * Compute dynamic retrieval depth based on query complexity
 *
 * Paper Reference: Section 3.3
 * k_dyn = ⌊k_base * (1 + δ * C_q)⌋
 *
 * @param baseK Base retrieval count (k_base)
 * @param complexity Query complexity score C_q ∈ [0, 1]
 * @param delta Expansion factor (default: 2.0)
 */
export function computeDynamicK(
  baseK: number,
  complexity: number,
  delta: number = 2.0,
): number {
  return Math.floor(baseK * (1 + delta * complexity));
}

/**
 * Compute hybrid retrieval score
 *
 * Paper Reference: Section 3.3
 * S(q, m_k) = α * cos(E(q), v_k) + β * BM25(q, m_k) + γ * 𝕀(constraints)
 *
 * @param semanticScore Semantic similarity (0-1)
 * @param lexicalScore BM25 score (normalized 0-1)
 * @param matchesConstraints Whether symbolic constraints are satisfied
 * @param alpha Semantic weight (default: 0.6)
 * @param beta Lexical weight (default: 0.3)
 * @param gamma Symbolic match bonus (default: 0.1)
 */
export function computeHybridScore(
  semanticScore: number,
  lexicalScore: number,
  matchesConstraints: boolean,
  alpha: number = 0.6,
  beta: number = 0.3,
  gamma: number = 0.1,
): number {
  const symbolicBonus = matchesConstraints ? gamma : 0;
  return alpha * semanticScore + beta * lexicalScore + symbolicBonus;
}
