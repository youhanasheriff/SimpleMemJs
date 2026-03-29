import { describe, it, expect } from "vitest";
import {
  parseTimestamp,
  parseTimeRange,
  getTimeDiffSeconds,
  isWithinRange,
  formatTimestamp,
  now,
  dayjs,
} from "./temporal";

// Fixed reference time for deterministic tests
const REF = "2025-06-15T12:00:00.000Z";

// =============================================================================
// parseTimestamp
// =============================================================================

describe("parseTimestamp", () => {
  it("parses ISO-8601 strings directly", () => {
    const result = parseTimestamp("2025-01-15T10:30:00.000Z");
    expect(result).toBe("2025-01-15T10:30:00.000Z");
  });

  it('returns ref time for "now"', () => {
    const result = parseTimestamp("now", REF);
    expect(result).toBe(REF);
  });

  it('returns start of ref day for "today"', () => {
    const result = parseTimestamp("today", REF);
    const expected = dayjs(REF).startOf("day").toISOString();
    expect(result).toBe(expected);
  });

  it('returns next day start for "tomorrow"', () => {
    const result = parseTimestamp("tomorrow", REF);
    const expected = dayjs(REF).add(1, "day").startOf("day").toISOString();
    expect(result).toBe(expected);
  });

  it('returns previous day start for "yesterday"', () => {
    const result = parseTimestamp("yesterday", REF);
    const expected = dayjs(REF).subtract(1, "day").startOf("day").toISOString();
    expect(result).toBe(expected);
  });

  it('handles "in X hours"', () => {
    const result = parseTimestamp("in 3 hours", REF);
    const expected = dayjs(REF).add(3, "hour").toISOString();
    expect(result).toBe(expected);
  });

  it('handles "in X days"', () => {
    const result = parseTimestamp("in 5 days", REF);
    const expected = dayjs(REF).add(5, "day").toISOString();
    expect(result).toBe(expected);
  });

  it('handles "X days ago"', () => {
    const result = parseTimestamp("2 days ago", REF);
    const expected = dayjs(REF).subtract(2, "day").toISOString();
    expect(result).toBe(expected);
  });

  it('handles "X hours ago"', () => {
    const result = parseTimestamp("6 hours ago", REF);
    const expected = dayjs(REF).subtract(6, "hour").toISOString();
    expect(result).toBe(expected);
  });

  it('handles "last week"', () => {
    const result = parseTimestamp("last week", REF);
    const expected = dayjs(REF)
      .subtract(1, "week")
      .startOf("week")
      .toISOString();
    expect(result).toBe(expected);
  });

  it('handles "last month"', () => {
    const result = parseTimestamp("last month", REF);
    const expected = dayjs(REF)
      .subtract(1, "month")
      .startOf("month")
      .toISOString();
    expect(result).toBe(expected);
  });

  it('handles "next week"', () => {
    const result = parseTimestamp("next week", REF);
    const expected = dayjs(REF).add(1, "week").startOf("week").toISOString();
    expect(result).toBe(expected);
  });

  it('handles "next month"', () => {
    const result = parseTimestamp("next month", REF);
    const expected = dayjs(REF).add(1, "month").startOf("month").toISOString();
    expect(result).toBe(expected);
  });

  it("parses YYYY-MM-DD format", () => {
    const result = parseTimestamp("2025-03-20");
    // dayjs parses dates in local timezone, so just verify it's valid and close
    expect(dayjs(result).isValid()).toBe(true);
    expect(dayjs(result).date()).toBe(20);
    expect(dayjs(result).month()).toBe(2); // March = 2
  });

  it("is case-insensitive for relative expressions", () => {
    const lower = parseTimestamp("Tomorrow", REF);
    const expected = dayjs(REF).add(1, "day").startOf("day").toISOString();
    expect(lower).toBe(expected);
  });

  it("returns original input for unparseable strings", () => {
    const result = parseTimestamp("not a real date at all xyz");
    expect(result).toBe("not a real date at all xyz");
  });
});

// =============================================================================
// parseTimeRange
// =============================================================================

describe("parseTimeRange", () => {
  it('parses "last week"', () => {
    const { start, end } = parseTimeRange("last week", REF);
    const expectedStart = dayjs(REF)
      .subtract(1, "week")
      .startOf("week")
      .toISOString();
    const expectedEnd = dayjs(REF)
      .subtract(1, "week")
      .endOf("week")
      .toISOString();
    expect(start).toBe(expectedStart);
    expect(end).toBe(expectedEnd);
  });

  it('parses "last month"', () => {
    const { start, end } = parseTimeRange("last month", REF);
    const expectedStart = dayjs(REF)
      .subtract(1, "month")
      .startOf("month")
      .toISOString();
    const expectedEnd = dayjs(REF)
      .subtract(1, "month")
      .endOf("month")
      .toISOString();
    expect(start).toBe(expectedStart);
    expect(end).toBe(expectedEnd);
  });

  it('parses "this week"', () => {
    const { start, end } = parseTimeRange("this week", REF);
    const expectedStart = dayjs(REF).startOf("week").toISOString();
    const expectedEnd = dayjs(REF).endOf("week").toISOString();
    expect(start).toBe(expectedStart);
    expect(end).toBe(expectedEnd);
  });

  it('parses "this month"', () => {
    const { start, end } = parseTimeRange("this month", REF);
    const expectedStart = dayjs(REF).startOf("month").toISOString();
    const expectedEnd = dayjs(REF).endOf("month").toISOString();
    expect(start).toBe(expectedStart);
    expect(end).toBe(expectedEnd);
  });

  it('parses "november 2025"', () => {
    const { start, end } = parseTimeRange("november 2025");
    const startDate = dayjs(start);
    const endDate = dayjs(end);
    expect(startDate.month()).toBe(10); // November = 10
    expect(startDate.year()).toBe(2025);
    expect(endDate.month()).toBe(10);
    expect(endDate.isAfter(startDate)).toBe(true);
  });

  it("parses a single date as full day range", () => {
    const { start, end } = parseTimeRange("2025-06-15");
    const startDate = dayjs(start);
    const endDate = dayjs(end);
    // Should be within June 15 in local timezone
    expect(startDate.isValid()).toBe(true);
    expect(endDate.isValid()).toBe(true);
    expect(endDate.isAfter(startDate)).toBe(true);
  });

  it("returns distant past to far future for unparseable input", () => {
    const { start, end } = parseTimeRange("gibberish xyz not a date");
    // start should be epoch, end should be far future
    expect(dayjs(start).year()).toBeLessThanOrEqual(1970);
    expect(dayjs(end).year()).toBeGreaterThan(2100);
  });
});

// =============================================================================
// getTimeDiffSeconds
// =============================================================================

describe("getTimeDiffSeconds", () => {
  it("returns 0 for identical timestamps", () => {
    expect(getTimeDiffSeconds(REF, REF)).toBe(0);
  });

  it("returns absolute difference in seconds", () => {
    const a = "2025-01-01T00:00:00.000Z";
    const b = "2025-01-01T01:00:00.000Z";
    expect(getTimeDiffSeconds(a, b)).toBe(3600);
  });

  it("is symmetric (absolute value)", () => {
    const a = "2025-01-01T00:00:00.000Z";
    const b = "2025-01-01T01:00:00.000Z";
    expect(getTimeDiffSeconds(a, b)).toBe(getTimeDiffSeconds(b, a));
  });

  it("returns Infinity for invalid timestamps", () => {
    expect(getTimeDiffSeconds("not a date", REF)).toBe(Infinity);
    expect(getTimeDiffSeconds(REF, "also not a date")).toBe(Infinity);
  });
});

// =============================================================================
// isWithinRange
// =============================================================================

describe("isWithinRange", () => {
  const start = "2025-01-01T00:00:00.000Z";
  const end = "2025-12-31T23:59:59.999Z";

  it("returns true for timestamp within range", () => {
    expect(isWithinRange("2025-06-15T12:00:00.000Z", start, end)).toBe(true);
  });

  it("returns true for timestamp at start boundary (inclusive)", () => {
    expect(isWithinRange(start, start, end)).toBe(true);
  });

  it("returns true for timestamp at end boundary (inclusive)", () => {
    expect(isWithinRange(end, start, end)).toBe(true);
  });

  it("returns false for timestamp before range", () => {
    expect(isWithinRange("2024-12-31T23:59:59.999Z", start, end)).toBe(false);
  });

  it("returns false for timestamp after range", () => {
    expect(isWithinRange("2026-01-01T00:00:00.000Z", start, end)).toBe(false);
  });

  it("returns false for invalid timestamp", () => {
    expect(isWithinRange("not a date", start, end)).toBe(false);
  });
});

// =============================================================================
// formatTimestamp
// =============================================================================

describe("formatTimestamp", () => {
  it("formats ISO timestamp to readable string", () => {
    const result = formatTimestamp("2025-11-16T14:00:00.000Z");
    // formatTimestamp uses local timezone, so just verify the format pattern
    expect(result).toMatch(/\d{1,2} November 2025 at \d{1,2}:\d{2} [AP]M/);
  });

  it("returns original string for invalid input", () => {
    expect(formatTimestamp("not a date")).toBe("not a date");
  });
});

// =============================================================================
// now
// =============================================================================

describe("now", () => {
  it("returns a valid ISO-8601 string", () => {
    const result = now();
    expect(dayjs(result).isValid()).toBe(true);
  });

  it("returns a time close to current time", () => {
    const before = Date.now();
    const result = now();
    const after = Date.now();
    const resultMs = dayjs(result).valueOf();
    expect(resultMs).toBeGreaterThanOrEqual(before);
    expect(resultMs).toBeLessThanOrEqual(after);
  });
});
