/**
 * LoCoMo-10 Benchmark Types
 * Mirrors the data structure of the LoCoMo-10 dataset
 */

export interface QA {
  question: string;
  answer?: string; // Ground truth answer
  evidence?: string[]; // IDs of turns providing evidence
  category?: number; // 0-5 (e.g., 5 is adversarial)
  adversarial_answer?: string;
}

export interface Turn {
  speaker: string;
  dia_id: string;
  text: string;
}

export interface Session {
  session_id: number;
  date_time: string; // ISO 8601
  turns: Turn[];
}

export interface Conversation {
  speaker_a: string;
  speaker_b: string;
  sessions: Record<number, Session>;
}

export interface EventSummary {
  events: Record<string, Record<string, string[]>>; // session -> speaker -> events
}

export interface Observation {
  observations: Record<string, Record<string, string[][]>>; // session -> speaker -> [observation, evidence]
}

export interface LoCoMoSample {
  sample_id: string;
  qa: QA[];
  conversation: Conversation;
  event_summary: EventSummary;
  observation: Observation;
  session_summary: Record<string, string>;
}

export interface BenchmarkResult {
  sample_id: string;
  question: string;
  prediction: string;
  ground_truth?: string;
  retrieval_time_ms: number;
  answer_time_ms: number;
  total_time_ms: number;
  metrics: {
    exact_match: number;
    f1_score: number;
    jaccard_similarity?: number;
    llm_judge_score?: number;
  };
}
