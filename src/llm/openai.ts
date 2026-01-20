/**
 * OpenAI-Compatible LLM Provider
 *
 * Works with OpenAI API, OpenRouter, Azure OpenAI, or any OpenAI-compatible endpoint
 */

import OpenAI from "openai";
import { z } from "zod";
import type { LLMProvider, LLMCompletionOptions } from "../types/index.js";

export interface OpenAIProviderOptions {
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
   * Model to use for completions
   * @default 'gpt-4o-mini'
   */
  model?: string;

  /**
   * Default temperature for completions
   * @default 0.1
   */
  defaultTemperature?: number;

  /**
   * Default max tokens
   * @default 4096
   */
  defaultMaxTokens?: number;
}

/**
 * OpenAI-compatible LLM provider
 */
export class OpenAIProvider implements LLMProvider {
  private client: OpenAI;
  private model: string;
  private defaultTemperature: number;
  private defaultMaxTokens: number;

  constructor(options: OpenAIProviderOptions) {
    this.client = new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
    });
    this.model = options.model ?? "gpt-4o-mini";
    this.defaultTemperature = options.defaultTemperature ?? 0.1;
    this.defaultMaxTokens = options.defaultMaxTokens ?? 4096;
  }

  async complete(
    prompt: string,
    options?: LLMCompletionOptions,
  ): Promise<string> {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    if (options?.systemPrompt) {
      messages.push({ role: "system", content: options.systemPrompt });
    }
    messages.push({ role: "user", content: prompt });

    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
      temperature: options?.temperature ?? this.defaultTemperature,
      max_tokens: options?.maxTokens ?? this.defaultMaxTokens,
    });

    return response.choices[0]?.message?.content ?? "";
  }

  async completeJSON<T>(prompt: string, schema: z.ZodType<T>): Promise<T> {
    const systemPrompt = `You are a helpful assistant that responds only with valid JSON. Do not include any text outside of the JSON object. Do not wrap the response in markdown code blocks.`;

    const response = await this.complete(prompt, {
      systemPrompt,
      temperature: 0.1, // Lower temperature for structured output
    });

    // Clean up response - remove markdown code blocks if present
    let cleaned = response.trim();
    if (cleaned.startsWith("```json")) {
      cleaned = cleaned.slice(7);
    } else if (cleaned.startsWith("```")) {
      cleaned = cleaned.slice(3);
    }
    if (cleaned.endsWith("```")) {
      cleaned = cleaned.slice(0, -3);
    }
    cleaned = cleaned.trim();

    // Parse and validate
    const parsed = JSON.parse(cleaned);
    return schema.parse(parsed);
  }
}
