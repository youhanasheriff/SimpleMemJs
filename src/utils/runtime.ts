/**
 * Runtime detection utilities for cross-platform compatibility
 *
 * Detects whether code is running in Bun, Deno, or Node.js
 */

export type Runtime = "bun" | "deno" | "node" | "browser" | "unknown";

/**
 * Detect the current JavaScript runtime
 */
export function detectRuntime(): Runtime {
  // Check for Bun
  if (typeof globalThis !== "undefined" && "Bun" in globalThis) {
    return "bun";
  }

  // Check for Deno
  if (typeof globalThis !== "undefined" && "Deno" in globalThis) {
    return "deno";
  }

  // Check for browser
  if (typeof window !== "undefined" && typeof window.document !== "undefined") {
    return "browser";
  }

  // Check for Node.js
  if (
    typeof process !== "undefined" &&
    process.versions &&
    process.versions.node
  ) {
    return "node";
  }

  return "unknown";
}

/**
 * Check if running in a server environment (not browser)
 */
export function isServer(): boolean {
  const runtime = detectRuntime();
  return runtime === "bun" || runtime === "deno" || runtime === "node";
}

/**
 * Check if file system APIs are available
 */
export function hasFileSystem(): boolean {
  const runtime = detectRuntime();
  return runtime === "bun" || runtime === "deno" || runtime === "node";
}

/**
 * Get runtime-specific fetch implementation
 * All modern runtimes have native fetch, but this provides a consistent interface
 */
export function getFetch(): typeof fetch {
  if (typeof fetch !== "undefined") {
    return fetch;
  }
  throw new Error("fetch is not available in this environment");
}

/**
 * Runtime information object
 */
export interface RuntimeInfo {
  runtime: Runtime;
  version: string;
  isServer: boolean;
  hasFileSystem: boolean;
}

/**
 * Get detailed runtime information
 */
export function getRuntimeInfo(): RuntimeInfo {
  const runtime = detectRuntime();
  let version = "unknown";
  const g = globalThis as any;

  switch (runtime) {
    case "bun":
      version = g.Bun?.version ?? "unknown";
      break;
    case "deno":
      version = g.Deno?.version?.deno ?? "unknown";
      break;
    case "node":
      version = process.versions.node;
      break;
    case "browser":
      version =
        typeof navigator !== "undefined" ? navigator.userAgent : "unknown";
      break;
  }

  return {
    runtime,
    version,
    isServer: isServer(),
    hasFileSystem: hasFileSystem(),
  };
}
