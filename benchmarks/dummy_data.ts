/**
 * Synthetic Dataset Generator for Benchmarking
 * Creates realistic conversation patterns to test SimpleMem without external datasets
 */

import { v4 as uuidv4 } from "uuid";
import type { LoCoMoSample, Conversation, Session, Turn, QA } from "./types";
import { now } from "../src/utils/temporal";

export function generateDummyData(numSamples: number = 1): LoCoMoSample[] {
  const samples: LoCoMoSample[] = [];

  for (let i = 0; i < numSamples; i++) {
    samples.push(createSingleSample(i));
  }

  return samples;
}

function createSingleSample(id: number): LoCoMoSample {
  const speakerA = "Alice";
  const speakerB = "Bob";
  const baseTime = "2025-06-15T10:00:00";

  // Create a multi-session conversation
  const session1 = createSession(1, baseTime, [
    {
      speaker: speakerA,
      text: "Hey Bob, are we still on for the marketing kickoff tomorrow?",
    },
    {
      speaker: speakerB,
      text: "Yes, definitely. I've booked Conference Room B at 2 PM.",
    },
    {
      speaker: speakerA,
      text: "Perfect. I'll make sure to bring the Q2 analytics report.",
    },
    {
      speaker: speakerB,
      text: "Great. Also, did you hear that Sarah got promoted to VP?",
    },
    {
      speaker: speakerA,
      text: "I did! She was promoted for leading the Atlas project.",
    },
  ]);

  const session2 = createSession(2, "2025-06-20T14:30:00", [
    {
      speaker: speakerB,
      text: "Alice, do you remember where we put the archive drives?",
    },
    {
      speaker: speakerA,
      text: "I think I left them in the secure cabinet in the server room.",
    },
    {
      speaker: speakerB,
      text: "Thanks. By the way, the client meeting has been moved to next Friday, June 27th.",
    },
    { speaker: speakerA, text: "Noted. Is it still at 10 AM?" },
    { speaker: speakerB, text: "No, it's pushed to 1 PM." },
  ]);

  const conversation: Conversation = {
    speaker_a: speakerA,
    speaker_b: speakerB,
    sessions: {
      1: session1,
      2: session2,
    },
  };

  const qa: QA[] = [
    {
      question: "When and where is the marketing kickoff?",
      answer: "16 June 2025 at 2:00 PM in Conference Room B",
      category: 1,
    },
    {
      question: "What will Alice bring to the kickoff?",
      answer: "Q2 analytics report",
      category: 1,
    },
    {
      question: "Why did Sarah get promoted?",
      answer: "For leading the Atlas project",
      category: 2,
    },
    {
      question: "Where are the archive drives located?",
      answer: "Secure cabinet in the server room",
      category: 1,
    },
    {
      question: "What time is the rescheduled client meeting?",
      answer: "27 June 2025 at 1:00 PM",
      category: 3, // Temporal update
    },
  ];

  return {
    sample_id: id.toString(),
    qa,
    conversation,
    event_summary: { events: {} },
    observation: { observations: {} },
    session_summary: {},
  };
}

function createSession(
  id: number,
  time: string,
  turnsData: { speaker: string; text: string }[],
): Session {
  const turns: Turn[] = turnsData.map((t, idx) => ({
    speaker: t.speaker,
    dia_id: `${id}_${idx}`,
    text: t.text,
  }));

  return {
    session_id: id,
    date_time: time,
    turns,
  };
}
