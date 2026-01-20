/**
 * Temporal utilities for date parsing and manipulation
 *
 * Uses dayjs for date handling as specified by user
 */

import dayjs from "dayjs";
import utc from "dayjs/plugin/utc.js";
import timezone from "dayjs/plugin/timezone.js";
import customParseFormat from "dayjs/plugin/customParseFormat.js";
import relativeTime from "dayjs/plugin/relativeTime.js";

// Extend dayjs with plugins
dayjs.extend(utc);
dayjs.extend(timezone);
dayjs.extend(customParseFormat);
dayjs.extend(relativeTime);

export { dayjs };

/**
 * Parse a timestamp string to ISO-8601 format
 *
 * @param input Timestamp string (various formats)
 * @param referenceTime Reference time for relative expressions
 * @returns ISO-8601 formatted string
 */
export function parseTimestamp(
  input: string,
  referenceTime?: Date | string,
): string {
  const ref = referenceTime ? dayjs(referenceTime) : dayjs();

  // Try direct ISO parsing first
  const direct = dayjs(input);
  if (direct.isValid()) {
    return direct.toISOString();
  }

  // Handle common relative expressions
  const lower = input.toLowerCase().trim();

  if (lower === "now") {
    return ref.toISOString();
  }

  if (lower === "today") {
    return ref.startOf("day").toISOString();
  }

  if (lower === "tomorrow") {
    return ref.add(1, "day").startOf("day").toISOString();
  }

  if (lower === "yesterday") {
    return ref.subtract(1, "day").startOf("day").toISOString();
  }

  // "in X hours/days/weeks"
  const inMatch = lower.match(/^in\s+(\d+)\s+(hour|day|week|month|year)s?$/);
  if (inMatch) {
    const [, amount, unit] = inMatch;
    return ref
      .add(parseInt(amount), unit as dayjs.ManipulateType)
      .toISOString();
  }

  // "X hours/days ago"
  const agoMatch = lower.match(/^(\d+)\s+(hour|day|week|month|year)s?\s+ago$/);
  if (agoMatch) {
    const [, amount, unit] = agoMatch;
    return ref
      .subtract(parseInt(amount), unit as dayjs.ManipulateType)
      .toISOString();
  }

  // "last week/month"
  const lastMatch = lower.match(/^last\s+(week|month|year)$/);
  if (lastMatch) {
    const [, unit] = lastMatch;
    return ref
      .subtract(1, unit as dayjs.ManipulateType)
      .startOf(unit as dayjs.ManipulateType)
      .toISOString();
  }

  // "next week/month"
  const nextMatch = lower.match(/^next\s+(week|month|year)$/);
  if (nextMatch) {
    const [, unit] = nextMatch;
    return ref
      .add(1, unit as dayjs.ManipulateType)
      .startOf(unit as dayjs.ManipulateType)
      .toISOString();
  }

  // Try common date formats
  const formats = [
    "YYYY-MM-DD",
    "MM/DD/YYYY",
    "DD/MM/YYYY",
    "MMMM D, YYYY",
    "MMM D, YYYY",
    "D MMMM YYYY",
    "YYYY-MM-DDTHH:mm:ss",
    "YYYY-MM-DD HH:mm:ss",
  ];

  for (const fmt of formats) {
    const parsed = dayjs(input, fmt, true);
    if (parsed.isValid()) {
      return parsed.toISOString();
    }
  }

  // Fallback: return original input (LLM should have normalized it)
  return input;
}

/**
 * Parse a time range expression
 *
 * @param expression Time range expression (e.g., "last week", "November 2025")
 * @param referenceTime Reference time for relative expressions
 * @returns Start and end ISO-8601 timestamps
 */
export function parseTimeRange(
  expression: string,
  referenceTime?: Date | string,
): { start: string; end: string } {
  const ref = referenceTime ? dayjs(referenceTime) : dayjs();
  const lower = expression.toLowerCase().trim();

  // "last week"
  if (lower === "last week") {
    const lastWeekStart = ref.subtract(1, "week").startOf("week");
    const lastWeekEnd = ref.subtract(1, "week").endOf("week");
    return {
      start: lastWeekStart.toISOString(),
      end: lastWeekEnd.toISOString(),
    };
  }

  // "last month"
  if (lower === "last month") {
    const lastMonthStart = ref.subtract(1, "month").startOf("month");
    const lastMonthEnd = ref.subtract(1, "month").endOf("month");
    return {
      start: lastMonthStart.toISOString(),
      end: lastMonthEnd.toISOString(),
    };
  }

  // "this week"
  if (lower === "this week") {
    return {
      start: ref.startOf("week").toISOString(),
      end: ref.endOf("week").toISOString(),
    };
  }

  // "this month"
  if (lower === "this month") {
    return {
      start: ref.startOf("month").toISOString(),
      end: ref.endOf("month").toISOString(),
    };
  }

  // Month + Year (e.g., "November 2025")
  const monthYearMatch = lower.match(
    /^(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})$/,
  );
  if (monthYearMatch) {
    const [, monthName, year] = monthYearMatch;
    const monthDate = dayjs(`${monthName} 1, ${year}`, "MMMM D, YYYY");
    return {
      start: monthDate.startOf("month").toISOString(),
      end: monthDate.endOf("month").toISOString(),
    };
  }

  // Single date - entire day
  const singleDate = dayjs(expression);
  if (singleDate.isValid()) {
    return {
      start: singleDate.startOf("day").toISOString(),
      end: singleDate.endOf("day").toISOString(),
    };
  }

  // Fallback: return distant past to far future
  return {
    start: dayjs(0).toISOString(),
    end: dayjs().add(100, "year").toISOString(),
  };
}

/**
 * Get time difference in seconds between two timestamps
 *
 * @param a First timestamp (ISO-8601)
 * @param b Second timestamp (ISO-8601)
 * @returns Absolute difference in seconds
 */
export function getTimeDiffSeconds(a: string, b: string): number {
  const timeA = dayjs(a);
  const timeB = dayjs(b);

  if (!timeA.isValid() || !timeB.isValid()) {
    return Infinity; // Unknown time = infinite difference
  }

  return Math.abs(timeA.diff(timeB, "second"));
}

/**
 * Check if a timestamp falls within a range
 *
 * @param timestamp Timestamp to check
 * @param start Range start (inclusive)
 * @param end Range end (inclusive)
 */
export function isWithinRange(
  timestamp: string,
  start: string,
  end: string,
): boolean {
  const time = dayjs(timestamp);
  const startTime = dayjs(start);
  const endTime = dayjs(end);

  if (!time.isValid()) {
    return false;
  }

  return (
    (time.isAfter(startTime) || time.isSame(startTime)) &&
    (time.isBefore(endTime) || time.isSame(endTime))
  );
}

/**
 * Format a timestamp for display (DD Month YYYY at HH:MM)
 */
export function formatTimestamp(timestamp: string): string {
  const time = dayjs(timestamp);
  if (!time.isValid()) {
    return timestamp;
  }
  return time.format("D MMMM YYYY [at] h:mm A");
}

/**
 * Get current timestamp in ISO-8601 format
 */
export function now(): string {
  return dayjs().toISOString();
}
