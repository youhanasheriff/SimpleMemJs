/**
 * LoCoMo-10 Benchmark Runner
 */

import { SimpleMem } from "../src/SimpleMem";
import type { SimpleMemOptions } from "../src/SimpleMem";
import type { LoCoMoSample, BenchmarkResult } from "./types";
import { generateDummyData } from "./dummy_data";
import {
  calculateExactMatch,
  calculateF1Score,
  calculateJaccardSimilarity,
  calculateLLMJudgeScore,
} from "./metrics";
import { performance } from "perf_hooks";
import fs from "fs/promises";

export class LoCoMoTester {
  private memOptions: SimpleMemOptions;

  constructor(options: SimpleMemOptions) {
    this.memOptions = options;
  }

  async loadDataset(path?: string, limit?: number): Promise<LoCoMoSample[]> {
    let samples: LoCoMoSample[];

    if (path) {
      console.log(`Loading dataset from ${path}...`);
      const content = await fs.readFile(path, "utf-8");
      samples = JSON.parse(content);
    } else {
      console.log("Generating synthetic dataset...");
      samples = generateDummyData(limit || 5);
    }

    if (limit && limit < samples.length) {
      samples = samples.slice(0, limit);
    }

    return samples;
  }

  async testSample(sample: LoCoMoSample): Promise<BenchmarkResult[]> {
    console.log(`\nTesting Sample ${sample.sample_id}...`);

    // Initialize fresh SimpleMem instance
    const memory = new SimpleMem(this.memOptions);
    await memory.clear();

    // 1. Add Dialogues
    const dialogues = Object.values(sample.conversation.sessions).flatMap(
      (session) =>
        session.turns.map((turn) => ({
          speaker: turn.speaker,
          content: turn.text,
          timestamp: session.date_time,
        })),
    );

    const startIngest = performance.now();
    await memory.addDialogues(dialogues);
    await memory.finalize();
    const endIngest = performance.now();
    console.log(
      `  Ingestion: ${dialogues.length} turns in ${(endIngest - startIngest).toFixed(2)}ms`,
    );

    // 2. Run Queries
    const results: BenchmarkResult[] = [];

    for (const qa of sample.qa) {
      if (qa.category === 5) continue; // Skip adversarial for basic benchmark

      const startQuery = performance.now();
      const prediction = await memory.ask(qa.question);
      const endQuery = performance.now();

      const groundTruth = qa.answer || "";
      const em = calculateExactMatch(prediction, groundTruth);
      const f1 = calculateF1Score(prediction, groundTruth);
      const jaccard = calculateJaccardSimilarity(prediction, groundTruth);

      // LLM Judge (using the same LLM provider as the memory system)
      const judgeResult = await calculateLLMJudgeScore(
        this.memOptions.llm,
        qa.question,
        prediction,
        groundTruth,
      );

      console.log(`  [Q] ${qa.question}`);
      console.log(`  [A] ${prediction}`);
      console.log(`  [Truth] ${groundTruth}`);
      console.log(
        `  Metrics - F1: ${f1.toFixed(2)}, EM: ${em}, Jaccard: ${jaccard.toFixed(2)}, Judge: ${judgeResult.score}`,
      );

      results.push({
        sample_id: sample.sample_id,
        question: qa.question,
        prediction,
        ground_truth: groundTruth,
        retrieval_time_ms: 0,
        answer_time_ms: endQuery - startQuery,
        total_time_ms: endQuery - startQuery,
        metrics: {
          exact_match: em,
          f1_score: f1,
          jaccard_similarity: jaccard,
          llm_judge_score: judgeResult.score,
        },
      });
    }

    return results;
  }

  async run(options: {
    datasetPath?: string;
    limit?: number;
    outputFile?: string;
  }): Promise<void> {
    const samples = await this.loadDataset(options.datasetPath, options.limit);
    const allResults: BenchmarkResult[] = [];
    const timings: { ingest: number[] } = { ingest: [] };

    for (const sample of samples) {
      const results = await this.testSample(sample);
      allResults.push(...results);
    }

    // Calculate aggregates
    const avgF1 =
      allResults.reduce((sum, r) => sum + r.metrics.f1_score, 0) /
      allResults.length;
    const avgLatency =
      allResults.reduce((sum, r) => sum + r.total_time_ms, 0) /
      allResults.length;

    const report = {
      summary: {
        total_samples: samples.length,
        total_questions: allResults.length,
        average_f1: avgF1,
        average_latency_ms: avgLatency,
      },
      results: allResults,
    };

    console.log("\n--- Benchmark Summary ---");
    console.log(`Total Samples: ${samples.length}`);
    console.log(`Total Questions: ${allResults.length}`);
    console.log(`Average F1 Score: ${(avgF1 * 100).toFixed(2)}%`);
    console.log(`Average Latency: ${avgLatency.toFixed(2)}ms`);

    if (options.outputFile) {
      await fs.writeFile(options.outputFile, JSON.stringify(report, null, 2));
      console.log(`\nReport saved to ${options.outputFile}`);
    }
  }
}
