/**
 * OpenAI-Compatible Embedding Provider
 *
 * Works with OpenAI API, OpenRouter, Azure OpenAI, or any OpenAI-compatible endpoint
 */

import OpenAI from "openai";
import type { EmbeddingProvider } from "../types/index.js";

export interface OpenAIEmbeddingsOptions {
  /**
   * API key for authentication
   */
  apiKey: string;

  /**
   * Base URL for the API (for OpenRouter, Azure, or custom endpoints)
   * @default 'https://api.openai.com/v1'
   */
  baseURL?: string;

  /**
   * Model to use for embeddings
   * @default 'text-embedding-3-small'
   */
  model?: string;

  /**
   * Embedding dimensions (for models that support reduced dimensions)
   */
  dimensions?: number;
}

/**
 * OpenAI-compatible embedding provider
 */
export class OpenAIEmbeddings implements EmbeddingProvider {
  private client: OpenAI;
  private model: string;
  private _dimensions: number;

  constructor(options: OpenAIEmbeddingsOptions) {
    this.client = new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
    });
    this.model = options.model ?? "text-embedding-3-small";
    this._dimensions = options.dimensions ?? 1536; // Default for text-embedding-3-small
  }

  get dimensions(): number {
    return this._dimensions;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    const response = await this.client.embeddings.create({
      model: this.model,
      input: texts,
      dimensions: this._dimensions,
    });

    // Sort by index to ensure correct order
    const sorted = response.data.sort((a, b) => a.index - b.index);
    return sorted.map((item) => item.embedding);
  }
}
