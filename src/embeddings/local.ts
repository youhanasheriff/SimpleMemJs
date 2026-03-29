/**
 * Local Embedding Provider using @xenova/transformers
 *
 * Runs embedding models locally without API calls.
 * Requires @xenova/transformers as an optional peer dependency.
 */

import type { EmbeddingProvider } from "../types/index.js";

export interface LocalEmbeddingsOptions {
  /**
   * Hugging Face model ID
   * @default 'Xenova/all-MiniLM-L6-v2'
   */
  model?: string;

  /**
   * Use quantized model for faster inference
   * @default true
   */
  quantized?: boolean;

  /**
   * Directory to cache downloaded models
   */
  cacheDir?: string;
}

// Model name -> embedding dimensions
const KNOWN_DIMENSIONS: Record<string, number> = {
  "Xenova/all-MiniLM-L6-v2": 384,
  "Xenova/all-MiniLM-L12-v2": 384,
  "Xenova/all-mpnet-base-v2": 768,
  "Xenova/bge-small-en-v1.5": 384,
  "Xenova/bge-base-en-v1.5": 768,
  "Xenova/gte-small": 384,
  "Xenova/gte-base": 768,
};

/**
 * Local embedding provider using @xenova/transformers.
 * Models are downloaded and cached on first use.
 */
export class LocalEmbeddings implements EmbeddingProvider {
  private pipeline: any = null;
  private model: string;
  private quantized: boolean;
  private cacheDir?: string;
  private _dimensions: number;
  private initialized = false;

  constructor(options: LocalEmbeddingsOptions = {}) {
    this.model = options.model ?? "Xenova/all-MiniLM-L6-v2";
    this.quantized = options.quantized ?? true;
    this.cacheDir = options.cacheDir;
    this._dimensions = KNOWN_DIMENSIONS[this.model] ?? 0;
  }

  get dimensions(): number {
    return this._dimensions;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    await this.ensureInitialized();

    const results: number[][] = [];
    for (const text of texts) {
      const output = await this.pipeline(text, {
        pooling: "mean",
        normalize: true,
      });
      const embedding = Array.from(output.data as Float32Array);
      results.push(embedding);
    }

    // Set dimensions from first result if not known
    if (this._dimensions === 0 && results.length > 0) {
      this._dimensions = results[0].length;
    }

    return results;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;

    let pipelineFn: any;
    try {
      // Dynamic import — @xenova/transformers is an optional peer dep
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const mod = await (Function('return import("@xenova/transformers")')() as Promise<any>);
      pipelineFn = mod.pipeline ?? mod.default?.pipeline;
    } catch {
      throw new Error(
        "Failed to import @xenova/transformers. Install it with: npm install @xenova/transformers",
      );
    }

    if (!pipelineFn) {
      throw new Error(
        "Could not find pipeline function in @xenova/transformers",
      );
    }

    const pipelineOptions: Record<string, unknown> = {
      quantized: this.quantized,
    };
    if (this.cacheDir) {
      pipelineOptions.cache_dir = this.cacheDir;
    }

    this.pipeline = await pipelineFn(
      "feature-extraction",
      this.model,
      pipelineOptions,
    );

    this.initialized = true;
  }
}
