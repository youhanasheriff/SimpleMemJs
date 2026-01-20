/**
 * File-based Storage Adapter
 *
 * JSON file persistence - portable, no native dependencies
 * Works across Bun, Node.js, and Deno
 */

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import type {
  StorageAdapter,
  MemoryUnit,
  AbstractMemory,
  QueryFilter,
  ExportData,
} from "../types/index.js";
import { isWithinRange } from "../utils/temporal.js";

export interface FileStorageOptions {
  /**
   * Path to the JSON storage file
   */
  path: string;

  /**
   * Whether to auto-save after each write operation
   * @default true
   */
  autoSave?: boolean;

  /**
   * Pretty print JSON output
   * @default false
   */
  prettyPrint?: boolean;
}

/**
 * File-based storage adapter using JSON
 */
export class FileStorage implements StorageAdapter {
  private units: Map<string, MemoryUnit> = new Map();
  private abstracts: Map<string, AbstractMemory> = new Map();
  private filePath: string;
  private autoSave: boolean;
  private prettyPrint: boolean;
  private loaded = false;

  constructor(options: FileStorageOptions) {
    this.filePath = options.path;
    this.autoSave = options.autoSave ?? true;
    this.prettyPrint = options.prettyPrint ?? false;
  }

  /**
   * Load data from file (called lazily on first access)
   */
  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return;

    try {
      const content = await readFile(this.filePath, "utf-8");
      const data: ExportData = JSON.parse(content);

      for (const unit of data.units) {
        this.units.set(unit.id, unit);
      }
      for (const abstract of data.abstracts) {
        this.abstracts.set(abstract.id, abstract);
      }
    } catch (error) {
      // File doesn't exist yet - start with empty state
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
    }

    this.loaded = true;
  }

  /**
   * Persist data to file
   */
  private async persist(): Promise<void> {
    if (!this.autoSave) return;

    const data: ExportData = {
      units: Array.from(this.units.values()),
      abstracts: Array.from(this.abstracts.values()),
      version: "1.0.0",
      exportedAt: new Date().toISOString(),
    };

    // Ensure directory exists
    await mkdir(dirname(this.filePath), { recursive: true });

    const content = this.prettyPrint
      ? JSON.stringify(data, null, 2)
      : JSON.stringify(data);

    await writeFile(this.filePath, content, "utf-8");
  }

  async saveUnit(unit: MemoryUnit): Promise<void> {
    await this.ensureLoaded();
    this.units.set(unit.id, unit);
    await this.persist();
  }

  async saveUnits(units: MemoryUnit[]): Promise<void> {
    await this.ensureLoaded();
    for (const unit of units) {
      this.units.set(unit.id, unit);
    }
    await this.persist();
  }

  async getUnit(id: string): Promise<MemoryUnit | null> {
    await this.ensureLoaded();
    return this.units.get(id) ?? null;
  }

  async getAllUnits(): Promise<MemoryUnit[]> {
    await this.ensureLoaded();
    return Array.from(this.units.values());
  }

  async queryUnits(filter: QueryFilter): Promise<MemoryUnit[]> {
    await this.ensureLoaded();
    const results: MemoryUnit[] = [];

    for (const unit of this.units.values()) {
      if (this.matchesFilter(unit, filter)) {
        results.push(unit);
      }
    }

    return results;
  }

  async deleteUnit(id: string): Promise<void> {
    await this.ensureLoaded();
    this.units.delete(id);
    await this.persist();
  }

  async saveAbstract(abstract: AbstractMemory): Promise<void> {
    await this.ensureLoaded();
    this.abstracts.set(abstract.id, abstract);
    await this.persist();
  }

  async getAllAbstracts(): Promise<AbstractMemory[]> {
    await this.ensureLoaded();
    return Array.from(this.abstracts.values());
  }

  async clear(): Promise<void> {
    this.units.clear();
    this.abstracts.clear();
    await this.persist();
  }

  async export(): Promise<ExportData> {
    await this.ensureLoaded();
    return {
      units: Array.from(this.units.values()),
      abstracts: Array.from(this.abstracts.values()),
      version: "1.0.0",
      exportedAt: new Date().toISOString(),
    };
  }

  async import(data: ExportData): Promise<void> {
    for (const unit of data.units) {
      this.units.set(unit.id, unit);
    }
    for (const abstract of data.abstracts) {
      this.abstracts.set(abstract.id, abstract);
    }
    await this.persist();
  }

  /**
   * Force save to file (useful when autoSave is disabled)
   */
  async save(): Promise<void> {
    const originalAutoSave = this.autoSave;
    this.autoSave = true;
    await this.persist();
    this.autoSave = originalAutoSave;
  }

  /**
   * Check if a memory unit matches the filter criteria
   */
  private matchesFilter(unit: MemoryUnit, filter: QueryFilter): boolean {
    if (filter.persons && filter.persons.length > 0) {
      const hasMatchingPerson = filter.persons.some((person) =>
        unit.persons.some((p) =>
          p.toLowerCase().includes(person.toLowerCase()),
        ),
      );
      if (!hasMatchingPerson) return false;
    }

    if (filter.entities && filter.entities.length > 0) {
      const hasMatchingEntity = filter.entities.some((entity) =>
        unit.entities.some((e) =>
          e.toLowerCase().includes(entity.toLowerCase()),
        ),
      );
      if (!hasMatchingEntity) return false;
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
}
