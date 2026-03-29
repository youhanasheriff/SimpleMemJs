/**
 * Text chunking utilities for document ingestion
 */

/**
 * Split text into chunks at paragraph boundaries, respecting size limits.
 *
 * @param text The text to chunk
 * @param chunkSize Maximum characters per chunk (default 2000)
 * @param overlap Characters of overlap between chunks (default 200)
 * @returns Array of text chunks
 */
export function chunkText(
  text: string,
  chunkSize: number = 2000,
  overlap: number = 200,
): string[] {
  if (text.length <= chunkSize) {
    return [text.trim()].filter((t) => t.length > 0);
  }

  // Split on paragraph boundaries (double newlines)
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim().length > 0);

  const chunks: string[] = [];
  let current = "";

  for (const paragraph of paragraphs) {
    const trimmed = paragraph.trim();

    // If a single paragraph exceeds chunkSize, split it by sentences
    if (trimmed.length > chunkSize) {
      if (current.length > 0) {
        chunks.push(current.trim());
        current = "";
      }
      const sentenceChunks = splitLongText(trimmed, chunkSize, overlap);
      chunks.push(...sentenceChunks);
      continue;
    }

    // If adding this paragraph would exceed the limit, start a new chunk
    if (current.length + trimmed.length + 2 > chunkSize && current.length > 0) {
      chunks.push(current.trim());
      // Carry overlap from the end of the previous chunk
      if (overlap > 0 && current.length > overlap) {
        current = current.slice(-overlap) + "\n\n" + trimmed;
      } else {
        current = trimmed;
      }
    } else {
      current = current.length > 0 ? current + "\n\n" + trimmed : trimmed;
    }
  }

  if (current.trim().length > 0) {
    chunks.push(current.trim());
  }

  return chunks;
}

/**
 * Split a long text that has no paragraph breaks into chunks.
 * Tries to split on sentence boundaries.
 */
function splitLongText(
  text: string,
  chunkSize: number,
  overlap: number,
): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    let end = Math.min(start + chunkSize, text.length);

    // Try to find a sentence boundary (. ! ?) near the end
    if (end < text.length) {
      const lastSentenceEnd = text.lastIndexOf(". ", end);
      const lastExclamation = text.lastIndexOf("! ", end);
      const lastQuestion = text.lastIndexOf("? ", end);
      const bestBreak = Math.max(lastSentenceEnd, lastExclamation, lastQuestion);

      if (bestBreak > start + chunkSize / 2) {
        end = bestBreak + 2; // Include the punctuation and space
      }
    }

    chunks.push(text.slice(start, end).trim());
    start = end - overlap;

    // Prevent infinite loop
    if (start >= end) break;
  }

  return chunks.filter((c) => c.length > 0);
}
