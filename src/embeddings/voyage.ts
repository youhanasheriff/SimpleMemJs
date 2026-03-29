/**
 * Voyage AI Multimodal Embedding Provider
 *
 * Supports text and image embeddings via Voyage AI's REST API.
 * Uses native fetch — no SDK dependency.
 */

import type {
  MultimodalEmbeddingProvider,
  MultimodalInput,
} from "../types/index.js";

export interface VoyageEmbeddingsOptions {
  /**
   * Voyage AI API key
   */
  apiKey: string;

  /**
   * Model to use
   * @default 'voyage-multimodal-3'
   */
  model?: string;

  /**
   * Embedding dimensions
   * @default 1024
   */
  dimensions?: number;

  /**
   * API base URL
   * @default 'https://api.voyageai.com/v1'
   */
  baseURL?: string;
}

/**
 * Voyage AI multimodal embedding provider.
 * Supports text and image inputs in the same embedding space.
 */
export class VoyageEmbeddings implements MultimodalEmbeddingProvider {
  private apiKey: string;
  private model: string;
  private _dimensions: number;
  private baseURL: string;

  readonly supportsImages = true;

  constructor(options: VoyageEmbeddingsOptions) {
    this.apiKey = options.apiKey;
    this.model = options.model ?? "voyage-multimodal-3";
    this._dimensions = options.dimensions ?? 1024;
    this.baseURL = options.baseURL ?? "https://api.voyageai.com/v1";
  }

  get dimensions(): number {
    return this._dimensions;
  }

  /**
   * Embed text-only inputs (EmbeddingProvider interface)
   */
  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const inputs = texts.map((t) => ({ type: "text" as const, content: t }));
    return this.embedMultimodal(inputs);
  }

  /**
   * Embed multimodal inputs (text + images)
   */
  async embedMultimodal(inputs: MultimodalInput[]): Promise<number[][]> {
    if (inputs.length === 0) return [];

    const formattedInputs = inputs.map((input) => {
      if (input.type === "text") {
        return { content: input.content, type: "text" };
      }
      return {
        content: input.content,
        type: "image_base64",
      };
    });

    const response = await fetch(
      `${this.baseURL}/multimodalembeddings`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          inputs: formattedInputs,
          output_dimensions: this._dimensions,
        }),
      },
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Voyage AI API error (${response.status}): ${errorText}`,
      );
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    return data.data.map((item) => item.embedding);
  }
}
