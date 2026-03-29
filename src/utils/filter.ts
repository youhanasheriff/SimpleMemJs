/**
 * Shared filter logic for matching memory units against query filters
 */

import type { MemoryUnit, QueryFilter } from "../types/index.js";
import { isWithinRange } from "./temporal.js";

/**
 * Check if a memory unit matches the given filter criteria.
 *
 * All filters use AND logic -- a unit must satisfy every specified filter.
 * String comparisons are case-insensitive substrings.
 */
export function matchesFilter(unit: MemoryUnit, filter: QueryFilter): boolean {
  if (filter.persons && filter.persons.length > 0) {
    const hasMatch = filter.persons.some((person) =>
      unit.persons.some((p) =>
        p.toLowerCase().includes(person.toLowerCase()),
      ),
    );
    if (!hasMatch) return false;
  }

  if (filter.entities && filter.entities.length > 0) {
    const hasMatch = filter.entities.some((entity) =>
      unit.entities.some((e) =>
        e.toLowerCase().includes(entity.toLowerCase()),
      ),
    );
    if (!hasMatch) return false;
  }

  if (filter.timestampRange && unit.timestamp) {
    const { start, end } = filter.timestampRange;
    if (start && end && !isWithinRange(unit.timestamp, start, end)) {
      return false;
    }
  }

  if (filter.location && unit.location) {
    if (
      !unit.location.toLowerCase().includes(filter.location.toLowerCase())
    ) {
      return false;
    }
  }

  if (filter.topic && unit.topic) {
    if (!unit.topic.toLowerCase().includes(filter.topic.toLowerCase())) {
      return false;
    }
  }

  return true;
}
