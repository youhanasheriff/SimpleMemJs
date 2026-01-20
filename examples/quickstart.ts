/**
 * SimpleMem Quickstart Example
 *
 * To run this example:
 * 1. Set your OPENAI_API_KEY environment variable
 * 2. Run: bun run examples/quickstart.ts
 */

import { SimpleMem } from "../src/SimpleMem";
import { OpenAIProvider } from "../src/llm/openai";
import { OpenAIEmbeddings } from "../src/embeddings/openai";

async function main() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error("Error: OPENAI_API_KEY environment variable is required");
    process.exit(1);
  }

  console.log("🧠 Initializing SimpleMem...");

  // 1. Initialize with OpenAI providers
  // You can also use other providers by changing the baseURL
  const memory = new SimpleMem({
    llm: new OpenAIProvider({ apiKey }),
    embeddings: new OpenAIEmbeddings({ apiKey }),
    // Optional: Configure storage to save to a file
    // storage: new FileStorage({ path: './memory.json' }),
  });

  console.log("📝 Adding dialogues (Stage 1 Compression)...");

  // 2. Add some dialogue context
  // SimpleMem will process this using the sliding window and LLM extraction
  await memory.addDialogues([
    {
      speaker: "Alice",
      content:
        "I'm scheduling the project kickoff. How about next Tuesday at 2 PM?",
      timestamp: "2025-11-10T10:00:00", // Context: "next Tuesday" relative to this
    },
    {
      speaker: "Bob",
      content: "Tuesday works. I'll book the main conference room.",
      timestamp: "2025-11-10T10:05:00",
    },
    {
      speaker: "Alice",
      content: "Great. I'll invite the engineering team and Sarah from design.",
      timestamp: "2025-11-10T10:06:00",
    },
  ]);

  // Force processing of any remaining buffered dialogues
  await memory.finalize();

  console.log("\n🔍 Querying memory (Stage 3 Adaptive Retrieval)...");

  // 3. Ask questions
  const questions = [
    "When is the project kickoff meeting?",
    "Where will the meeting be held?",
    "Who is invited to the kickoff?",
  ];

  for (const q of questions) {
    console.log(`\n❓ Q: ${q}`);
    const answer = await memory.ask(q);
    console.log(`💡 A: ${answer}`);
  }

  console.log("\n✅ Demo complete!");
}

main().catch(console.error);
