/**
 * SQLite Storage Adapter
 *
 * Production-grade local persistence with ACID transactions,
 * indexed queries, and incremental writes.
 *
 * Uses bun:sqlite on Bun, better-sqlite3 on Node.js.
 */

import type {
  StorageAdapter,
  MemoryUnit,
  AbstractMemory,
  QueryFilter,
  ExportData,
} from "../types/index.js";
import { detectRuntime } from "../utils/runtime.js";

export interface SQLiteStorageOptions {
  /**
   * Path to the SQLite database file.
   * Use ':memory:' for in-memory database.
   */
  path: string;

  /**
   * Enable WAL mode for better concurrent read performance
   * @default true
   */
  walMode?: boolean;
}

/**
 * SQLite storage adapter with indexed queries and ACID transactions.
 */
export class SQLiteStorage implements StorageAdapter {
  private db: any;
  private initialized = false;
  private dbPath: string;
  private walMode: boolean;

  constructor(options: SQLiteStorageOptions) {
    this.dbPath = options.path;
    this.walMode = options.walMode ?? true;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;

    const runtime = detectRuntime();

    if (runtime === "bun") {
      const mod = await (Function('return import("bun:sqlite")')() as Promise<any>);
      const Database = mod.Database ?? mod.default?.Database ?? mod.default;
      this.db = new Database(this.dbPath);
    } else {
      try {
        const mod = await (Function('return import("better-sqlite3")')() as Promise<any>);
        const Database = mod.default ?? mod;
        this.db = new Database(this.dbPath);
      } catch {
        throw new Error(
          "Failed to import better-sqlite3. Install it with: npm install better-sqlite3",
        );
      }
    }

    if (this.walMode) {
      this.db.exec("PRAGMA journal_mode=WAL");
    }

    this.createTables();
    this.initialized = true;
  }

  private createTables(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memory_units (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        keywords TEXT NOT NULL DEFAULT '[]',
        timestamp TEXT,
        location TEXT,
        persons TEXT NOT NULL DEFAULT '[]',
        entities TEXT NOT NULL DEFAULT '[]',
        topic TEXT,
        salience TEXT NOT NULL DEFAULT 'medium',
        embedding TEXT,
        source_dialogue_ids TEXT NOT NULL DEFAULT '[]',
        created_at TEXT
      );

      CREATE TABLE IF NOT EXISTS abstract_memories (
        id TEXT PRIMARY KEY,
        pattern TEXT NOT NULL,
        source_unit_ids TEXT NOT NULL DEFAULT '[]',
        frequency INTEGER NOT NULL DEFAULT 1,
        first_occurrence TEXT NOT NULL,
        last_occurrence TEXT NOT NULL,
        entities TEXT NOT NULL DEFAULT '[]',
        embedding TEXT,
        is_archived INTEGER NOT NULL DEFAULT 0
      );

      CREATE INDEX IF NOT EXISTS idx_units_timestamp ON memory_units(timestamp);
      CREATE INDEX IF NOT EXISTS idx_units_location ON memory_units(location);
      CREATE INDEX IF NOT EXISTS idx_units_topic ON memory_units(topic);
      CREATE INDEX IF NOT EXISTS idx_units_salience ON memory_units(salience);
    `);
  }

  // ---------------------------------------------------------------------------
  // Serialization helpers
  // ---------------------------------------------------------------------------

  private serializeUnit(unit: MemoryUnit): Record<string, unknown> {
    return {
      id: unit.id,
      content: unit.content,
      keywords: JSON.stringify(unit.keywords),
      timestamp: unit.timestamp ?? null,
      location: unit.location ?? null,
      persons: JSON.stringify(unit.persons),
      entities: JSON.stringify(unit.entities),
      topic: unit.topic ?? null,
      salience: unit.salience,
      embedding: unit.embedding ? JSON.stringify(unit.embedding) : null,
      source_dialogue_ids: JSON.stringify(unit.sourceDialogueIds),
      created_at: unit.createdAt ?? null,
    };
  }

  private deserializeUnit(row: any): MemoryUnit {
    return {
      id: row.id,
      content: row.content,
      keywords: JSON.parse(row.keywords || "[]"),
      timestamp: row.timestamp ?? undefined,
      location: row.location ?? undefined,
      persons: JSON.parse(row.persons || "[]"),
      entities: JSON.parse(row.entities || "[]"),
      topic: row.topic ?? undefined,
      salience: row.salience,
      embedding: row.embedding ? JSON.parse(row.embedding) : undefined,
      sourceDialogueIds: JSON.parse(row.source_dialogue_ids || "[]"),
      createdAt: row.created_at ?? undefined,
    };
  }

  private serializeAbstract(abstract: AbstractMemory): Record<string, unknown> {
    return {
      id: abstract.id,
      pattern: abstract.pattern,
      source_unit_ids: JSON.stringify(abstract.sourceUnitIds),
      frequency: abstract.frequency,
      first_occurrence: abstract.firstOccurrence,
      last_occurrence: abstract.lastOccurrence,
      entities: JSON.stringify(abstract.entities),
      embedding: abstract.embedding
        ? JSON.stringify(abstract.embedding)
        : null,
      is_archived: abstract.isArchived ? 1 : 0,
    };
  }

  private deserializeAbstract(row: any): AbstractMemory {
    return {
      id: row.id,
      pattern: row.pattern,
      sourceUnitIds: JSON.parse(row.source_unit_ids || "[]"),
      frequency: row.frequency,
      firstOccurrence: row.first_occurrence,
      lastOccurrence: row.last_occurrence,
      entities: JSON.parse(row.entities || "[]"),
      embedding: row.embedding ? JSON.parse(row.embedding) : undefined,
      isArchived: row.is_archived === 1,
    };
  }

  // ---------------------------------------------------------------------------
  // StorageAdapter implementation
  // ---------------------------------------------------------------------------

  async saveUnit(unit: MemoryUnit): Promise<void> {
    await this.ensureInitialized();
    const data = this.serializeUnit(unit);
    this.db
      .prepare(
        `INSERT OR REPLACE INTO memory_units
         (id, content, keywords, timestamp, location, persons, entities, topic, salience, embedding, source_dialogue_ids, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        data.id, data.content, data.keywords, data.timestamp,
        data.location, data.persons, data.entities, data.topic,
        data.salience, data.embedding, data.source_dialogue_ids, data.created_at,
      );
  }

  async saveUnits(units: MemoryUnit[]): Promise<void> {
    await this.ensureInitialized();
    const stmt = this.db.prepare(
      `INSERT OR REPLACE INTO memory_units
       (id, content, keywords, timestamp, location, persons, entities, topic, salience, embedding, source_dialogue_ids, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );

    const insertMany = this.db.transaction((items: MemoryUnit[]) => {
      for (const unit of items) {
        const data = this.serializeUnit(unit);
        stmt.run(
          data.id, data.content, data.keywords, data.timestamp,
          data.location, data.persons, data.entities, data.topic,
          data.salience, data.embedding, data.source_dialogue_ids, data.created_at,
        );
      }
    });

    insertMany(units);
  }

  async getUnit(id: string): Promise<MemoryUnit | null> {
    await this.ensureInitialized();
    const row = this.db
      .prepare("SELECT * FROM memory_units WHERE id = ?")
      .get(id);
    return row ? this.deserializeUnit(row) : null;
  }

  async getAllUnits(): Promise<MemoryUnit[]> {
    await this.ensureInitialized();
    const rows = this.db
      .prepare("SELECT * FROM memory_units")
      .all();
    return rows.map((r: any) => this.deserializeUnit(r));
  }

  async queryUnits(filter: QueryFilter): Promise<MemoryUnit[]> {
    await this.ensureInitialized();

    // For complex JSON-based filtering, fetch all and filter in JS
    // This is pragmatic: SQLite's json_each requires careful SQL generation
    // and the in-memory filter is fast for typical agent workloads (<100K units)
    const allUnits = await this.getAllUnits();
    const { matchesFilter } = await import("../utils/filter.js");
    return allUnits.filter((unit) => matchesFilter(unit, filter));
  }

  async deleteUnit(id: string): Promise<void> {
    await this.ensureInitialized();
    this.db.prepare("DELETE FROM memory_units WHERE id = ?").run(id);
  }

  async saveAbstract(abstract: AbstractMemory): Promise<void> {
    await this.ensureInitialized();
    const data = this.serializeAbstract(abstract);
    this.db
      .prepare(
        `INSERT OR REPLACE INTO abstract_memories
         (id, pattern, source_unit_ids, frequency, first_occurrence, last_occurrence, entities, embedding, is_archived)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        data.id, data.pattern, data.source_unit_ids, data.frequency,
        data.first_occurrence, data.last_occurrence, data.entities,
        data.embedding, data.is_archived,
      );
  }

  async getAllAbstracts(): Promise<AbstractMemory[]> {
    await this.ensureInitialized();
    const rows = this.db
      .prepare("SELECT * FROM abstract_memories")
      .all();
    return rows.map((r: any) => this.deserializeAbstract(r));
  }

  async clear(): Promise<void> {
    await this.ensureInitialized();
    this.db.exec("DELETE FROM memory_units");
    this.db.exec("DELETE FROM abstract_memories");
  }

  async export(): Promise<ExportData> {
    await this.ensureInitialized();
    return {
      units: await this.getAllUnits(),
      abstracts: await this.getAllAbstracts(),
      version: "1.0.0",
      exportedAt: new Date().toISOString(),
    };
  }

  async import(data: ExportData): Promise<void> {
    await this.ensureInitialized();
    if (data.units.length > 0) {
      await this.saveUnits(data.units);
    }
    for (const abstract of data.abstracts) {
      await this.saveAbstract(abstract);
    }
  }

  /**
   * Close the database connection
   */
  close(): void {
    if (this.db) {
      this.db.close();
      this.initialized = false;
    }
  }
}
