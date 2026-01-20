/**
 * Benchmark Metrics Calculation
 */

import { LLMProvider } from "../src/types/index";
// Imports for other metrics would go here (e.g. natural, rogue, etc)
// For now we implement basic versions or placeholders where libraries are complex

/**
 * Normalize answer text for comparison
 * Lowercase, remove punctuation, extra whitespace
 */
export function normalizeAnswer(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .split(/\s+/)
    .join(" ")
    .trim();
}

/**
 * Calculate Exact Match score (0 or 1)
 */
export function calculateExactMatch(
  prediction: string,
  reference: string,
): number {
  return normalizeAnswer(prediction) === normalizeAnswer(reference) ? 1 : 0;
}

/**
 * Calculate token-level F1 score
 */
export function calculateF1Score(
  prediction: string,
  reference: string,
): number {
  const predTokens = normalizeAnswer(prediction).split(" ");
  const refTokens = normalizeAnswer(reference).split(" ");

  if (predTokens.length === 0 || refTokens.length === 0) {
    return predTokens.length === refTokens.length ? 1 : 0;
  }

  const common = predTokens.filter((t) => refTokens.includes(t));
  const numCommon = common.length;

  const precision = numCommon / predTokens.length;
  const recall = numCommon / refTokens.length;

  if (precision + recall === 0) return 0;
  return (2 * precision * recall) / (precision + recall);
}

/**
 * Simple Jaccard Similarity (placeholder for more complex semantic scores if libs unavailable)
 */
export function calculateJaccardSimilarity(
  prediction: string,
  reference: string,
): number {
  const predSet = new Set(normalizeAnswer(prediction).split(" "));
  const refSet = new Set(normalizeAnswer(reference).split(" "));

  const intersection = new Set([...predSet].filter((x) => refSet.has(x)));
  const union = new Set([...predSet, ...refSet]);

  return intersection.size / union.size;
}

/**
 * LLM Judge for semantic equivalence
 * Checks if the prediction conveys the same meaning as the ground truth
 */
export async function calculateLLMJudgeScore(
  llm: LLMProvider,
  question: string,
  prediction: string,
  reference: string,
): Promise<{ score: number; reasoning: string }> {
  const prompt = `You are an impartial judge evaluating the quality of an answer to a question.
  
Question: ${question}
Ground Truth: ${reference}
Prediction: ${prediction}

Compare the Prediction to the Ground Truth. Does the Prediction contain the core information required by the Ground Truth?
- Allow for different phrasing, synonyms, or levels of detail as long as the core fact is correct.
- If the Ground Truth is "Not mentioned", the Prediction must also state that content is missing or unknown.

Output JSON only:
{
  "score": <number between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}`;

  try {
    // Simple JSON parsing if provider doesn't support strict schema yet
    const response = await llm.complete(prompt);
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
    return { score: 0, reasoning: "Failed to parse LLM response" };
  } catch (e) {
    return { score: 0, reasoning: `Error: ${e}` };
  }
}
