/**
 * Benchmark CLI Entry Point
 * usage: bun run benchmarks/run.ts [options]
 */

import { OpenAIProvider } from "../src/llm/openai";
import { OpenAIEmbeddings } from "../src/embeddings/openai";
import { LoCoMoTester } from "./locomo";
import { parseArgs } from "util";

async function main() {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      dataset: { type: "string", short: "d" },
      output: { type: "string", short: "o", default: "benchmark_results.json" },
      limit: { type: "string", short: "n" }, // Parse as string then int
    },
  });

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error("Error: OPENAI_API_KEY environment variable is required");
    process.exit(1);
  }

  const memOptions = {
    llm: new OpenAIProvider({
      apiKey,
    }),
    embeddings: new OpenAIEmbeddings({
      apiKey,
    }),
  };

  const tester = new LoCoMoTester(memOptions);

  await tester.run({
    datasetPath: values.dataset,
    outputFile: values.output,
    limit: values.limit ? parseInt(values.limit) : undefined,
  });
}

main().catch(console.error);
