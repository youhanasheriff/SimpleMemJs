/**
 * Typed Event System for SimpleMem
 *
 * Cross-runtime compatible event emitter with no external dependencies.
 */

import type { MemoryUnit, AbstractMemory } from "./types/index.js";

// =============================================================================
// Event Type Definitions
// =============================================================================

export interface SimpleMemEvents {
  "memory:units_created": { units: MemoryUnit[]; count: number };
  "memory:units_indexed": { unitIds: string[]; count: number };
  "memory:abstract_created": {
    abstract: AbstractMemory;
    sourceCount: number;
  };
  "memory:consolidation_completed": {
    abstractsCreated: number;
    durationMs: number;
  };
  "retrieval:query_analyzed": { query: string; complexity: string };
  "retrieval:context_retrieved": {
    query: string;
    unitCount: number;
    abstractCount: number;
  };
  "retrieval:answer_generated": { query: string; answer: string };
  "storage:cleared": Record<string, never>;
  "error": { stage: string; error: Error; context?: string };
}

// =============================================================================
// Event Emitter
// =============================================================================

type EventHandler<T> = (data: T) => void;

export class SimpleMemEventEmitter {
  private handlers = new Map<string, Set<Function>>();

  /**
   * Subscribe to an event
   */
  on<K extends keyof SimpleMemEvents>(
    event: K,
    handler: EventHandler<SimpleMemEvents[K]>,
  ): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
  }

  /**
   * Unsubscribe from an event
   */
  off<K extends keyof SimpleMemEvents>(
    event: K,
    handler: EventHandler<SimpleMemEvents[K]>,
  ): void {
    this.handlers.get(event)?.delete(handler);
  }

  /**
   * Subscribe to an event (fires once then auto-unsubscribes)
   */
  once<K extends keyof SimpleMemEvents>(
    event: K,
    handler: EventHandler<SimpleMemEvents[K]>,
  ): void {
    const wrapper = (data: SimpleMemEvents[K]) => {
      this.off(event, wrapper);
      handler(data);
    };
    this.on(event, wrapper);
  }

  /**
   * Emit an event (synchronous, fire-and-forget)
   */
  emit<K extends keyof SimpleMemEvents>(
    event: K,
    data: SimpleMemEvents[K],
  ): void {
    const handlers = this.handlers.get(event);
    if (!handlers) return;
    for (const handler of handlers) {
      try {
        handler(data);
      } catch {
        // Event handlers should not break the pipeline
      }
    }
  }

  /**
   * Remove all handlers for a specific event, or all events
   */
  removeAllListeners(event?: keyof SimpleMemEvents): void {
    if (event) {
      this.handlers.delete(event);
    } else {
      this.handlers.clear();
    }
  }
}
