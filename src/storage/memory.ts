/**
 * In-Memory Storage Adapter
 *
 * Default storage using Maps - ephemeral, no persistence
 * Best for development and testing
 */

import type {
  StorageAdapter,
  MemoryUnit,
  AbstractMemory,
  QueryFilter,
  ExportData,
} from "../types/index.js";
import { isWithinRange } from "../utils/temporal.js";

/**
 * In-memory storage adapter using JavaScript Maps
 */
export class MemoryStorage implements StorageAdapter {
  private units: Map<string, MemoryUnit> = new Map();
  private abstracts: Map<string, AbstractMemory> = new Map();

  async saveUnit(unit: MemoryUnit): Promise<void> {
    this.units.set(unit.id, unit);
  }

  async saveUnits(units: MemoryUnit[]): Promise<void> {
    for (const unit of units) {
      this.units.set(unit.id, unit);
    }
  }

  async getUnit(id: string): Promise<MemoryUnit | null> {
    return this.units.get(id) ?? null;
  }

  async getAllUnits(): Promise<MemoryUnit[]> {
    return Array.from(this.units.values());
  }

  async queryUnits(filter: QueryFilter): Promise<MemoryUnit[]> {
    const results: MemoryUnit[] = [];

    for (const unit of this.units.values()) {
      if (this.matchesFilter(unit, filter)) {
        results.push(unit);
      }
    }

    return results;
  }

  async deleteUnit(id: string): Promise<void> {
    this.units.delete(id);
  }

  async saveAbstract(abstract: AbstractMemory): Promise<void> {
    this.abstracts.set(abstract.id, abstract);
  }

  async getAllAbstracts(): Promise<AbstractMemory[]> {
    return Array.from(this.abstracts.values());
  }

  async clear(): Promise<void> {
    this.units.clear();
    this.abstracts.clear();
  }

  async export(): Promise<ExportData> {
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
  }

  /**
   * Check if a memory unit matches the filter criteria
   */
  private matchesFilter(unit: MemoryUnit, filter: QueryFilter): boolean {
    // Check persons filter
    if (filter.persons && filter.persons.length > 0) {
      const hasMatchingPerson = filter.persons.some((person) =>
        unit.persons.some((p) =>
          p.toLowerCase().includes(person.toLowerCase()),
        ),
      );
      if (!hasMatchingPerson) return false;
    }

    // Check entities filter
    if (filter.entities && filter.entities.length > 0) {
      const hasMatchingEntity = filter.entities.some((entity) =>
        unit.entities.some((e) =>
          e.toLowerCase().includes(entity.toLowerCase()),
        ),
      );
      if (!hasMatchingEntity) return false;
    }

    // Check timestamp range
    if (filter.timestampRange && unit.timestamp) {
      const { start, end } = filter.timestampRange;
      if (start && end && !isWithinRange(unit.timestamp, start, end)) {
        return false;
      }
    }

    // Check location filter
    if (filter.location && unit.location) {
      if (
        !unit.location.toLowerCase().includes(filter.location.toLowerCase())
      ) {
        return false;
      }
    }

    // Check topic filter
    if (filter.topic && unit.topic) {
      if (!unit.topic.toLowerCase().includes(filter.topic.toLowerCase())) {
        return false;
      }
    }

    return true;
  }
}
