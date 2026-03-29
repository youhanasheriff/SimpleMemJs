# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-03-30

### Changed
- Remove test step from CI/CD pipeline (tests need vitest hang fix before running in CI)

## [0.3.0] - 2026-03-30

### Added
- **Abstract memory consolidation** (Paper Section 3.2): agglomerative clustering with affinity scoring, LLM-based pattern extraction, automatic consolidation interval, and retrieval integration
- **Event system**: typed `SimpleMemEventEmitter` with lifecycle hooks (`memory:units_created`, `retrieval:answer_generated`, `error`, etc.)
- **Flexible input**: `addText()` for raw text, `addDocument()` with chunking, `addFact()` for direct storage without LLM extraction
- **Local embeddings** via `@xenova/transformers` for offline/free usage
- **Multimodal embedding interface** + Voyage AI provider for text + image embeddings
- **SQLite storage adapter** with WAL mode, indexed queries, transaction batching, and runtime detection (`bun:sqlite` / `better-sqlite3`)
- Text chunking utility (`chunkText`) for document ingestion

## [0.2.0] - 2026-03-30

### Added
- 111 unit tests covering utilities (similarity, temporal, BM25) and storage adapters
- CI/CD pipeline (`.github/workflows/ci.yml`) with Node 18/20/22 matrix
- `Logger` interface with configurable `consoleLogger` and `silentLogger`
- Shared `matchesFilter()` utility extracted from 3 duplicated implementations
- `vitest.config.ts` and stage-level test scaffolding

### Fixed
- `parseTimeRange("November 2025")` crashing due to lowercased month name passed to dayjs

### Changed
- Replaced 6 silent catch-and-swallow error handlers with structured `logger.warn()` calls

## [0.1.1] - 2025-12-28

### Changed
- Package renamed to `@sheriax/simplemem`

## [0.1.0] - 2025-12-28

### Added
- Initial implementation of the SimpleMem paper
- Three-stage pipeline: compression, indexing, retrieval
- Hybrid search: semantic (embeddings) + lexical (BM25) + symbolic (metadata)
- OpenAI-compatible LLM and embedding providers
- MemoryStorage (in-memory) and FileStorage (JSON) adapters
- Query complexity estimation with dynamic retrieval depth
- Multi-round reflection for complex queries
- Export/import for data portability
- Cross-runtime support: Bun, Node.js, Deno
