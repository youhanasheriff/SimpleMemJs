import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { randomUUID } from "node:crypto";
import { rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { MemoryStorage } from "./memory";
import { FileStorage } from "./file";
import type { StorageAdapter, MemoryUnit, AbstractMemory } from "../types/index";

// =============================================================================
// Helpers
// =============================================================================

function makeUnit(overrides: Partial<MemoryUnit> = {}): MemoryUnit {
  return {
    id: randomUUID(),
    content: "Alice meets Bob at Starbucks on Monday",
    keywords: ["alice", "bob", "starbucks", "monday"],
    persons: ["Alice", "Bob"],
    entities: ["Starbucks"],
    sourceDialogueIds: [1, 2],
    salience: "medium",
    timestamp: "2025-06-15T14:00:00.000Z",
    location: "Starbucks Downtown",
    topic: "meeting",
    createdAt: "2025-06-15T12:00:00.000Z",
    ...overrides,
  };
}

function makeAbstract(overrides: Partial<AbstractMemory> = {}): AbstractMemory {
  return {
    id: randomUUID(),
    pattern: "Alice and Bob meet weekly at cafes",
    sourceUnitIds: [randomUUID(), randomUUID()],
    frequency: 5,
    firstOccurrence: "2025-01-01T00:00:00.000Z",
    lastOccurrence: "2025-06-01T00:00:00.000Z",
    entities: ["Starbucks"],
    isArchived: false,
    ...overrides,
  };
}

// =============================================================================
// Shared test suite for all StorageAdapter implementations
// =============================================================================

function testStorageAdapter(
  name: string,
  createStorage: () => StorageAdapter,
  cleanup?: () => Promise<void>,
) {
  describe(name, () => {
    let storage: StorageAdapter;

    beforeEach(() => {
      storage = createStorage();
    });

    afterEach(async () => {
      if (cleanup) await cleanup();
    });

    // --- Unit CRUD ---

    describe("unit CRUD", () => {
      it("saves and retrieves a unit", async () => {
        const unit = makeUnit();
        await storage.saveUnit(unit);
        const retrieved = await storage.getUnit(unit.id);
        expect(retrieved).not.toBeNull();
        expect(retrieved!.content).toBe(unit.content);
        expect(retrieved!.id).toBe(unit.id);
      });

      it("saves multiple units at once", async () => {
        const units = [makeUnit(), makeUnit(), makeUnit()];
        await storage.saveUnits(units);
        const all = await storage.getAllUnits();
        expect(all).toHaveLength(3);
      });

      it("returns null for non-existent unit", async () => {
        const result = await storage.getUnit(randomUUID());
        expect(result).toBeNull();
      });

      it("deletes a unit", async () => {
        const unit = makeUnit();
        await storage.saveUnit(unit);
        await storage.deleteUnit(unit.id);
        const result = await storage.getUnit(unit.id);
        expect(result).toBeNull();
      });

      it("getAllUnits returns all saved units", async () => {
        await storage.saveUnits([makeUnit(), makeUnit()]);
        const all = await storage.getAllUnits();
        expect(all).toHaveLength(2);
      });

      it("overwrites unit with same id", async () => {
        const id = randomUUID();
        await storage.saveUnit(makeUnit({ id, content: "original" }));
        await storage.saveUnit(makeUnit({ id, content: "updated" }));
        const result = await storage.getUnit(id);
        expect(result!.content).toBe("updated");
      });
    });

    // --- Query Filtering ---

    describe("queryUnits", () => {
      it("filters by persons", async () => {
        await storage.saveUnits([
          makeUnit({ persons: ["Alice"] }),
          makeUnit({ persons: ["Bob"] }),
          makeUnit({ persons: ["Charlie"] }),
        ]);
        const results = await storage.queryUnits({ persons: ["Alice"] });
        expect(results).toHaveLength(1);
        expect(results[0].persons).toContain("Alice");
      });

      it("filters by entities", async () => {
        await storage.saveUnits([
          makeUnit({ entities: ["Google"] }),
          makeUnit({ entities: ["Apple"] }),
        ]);
        const results = await storage.queryUnits({ entities: ["Google"] });
        expect(results).toHaveLength(1);
        expect(results[0].entities).toContain("Google");
      });

      it("filters by location (case-insensitive substring)", async () => {
        await storage.saveUnits([
          makeUnit({ location: "Starbucks Downtown" }),
          makeUnit({ location: "Central Park" }),
        ]);
        const results = await storage.queryUnits({ location: "starbucks" });
        expect(results).toHaveLength(1);
      });

      it("filters by topic (case-insensitive substring)", async () => {
        await storage.saveUnits([
          makeUnit({ topic: "project meeting" }),
          makeUnit({ topic: "lunch" }),
        ]);
        const results = await storage.queryUnits({ topic: "meeting" });
        expect(results).toHaveLength(1);
      });

      it("filters by timestamp range", async () => {
        await storage.saveUnits([
          makeUnit({ timestamp: "2025-03-01T00:00:00.000Z" }),
          makeUnit({ timestamp: "2025-06-15T00:00:00.000Z" }),
          makeUnit({ timestamp: "2025-09-01T00:00:00.000Z" }),
        ]);
        const results = await storage.queryUnits({
          timestampRange: {
            start: "2025-05-01T00:00:00.000Z",
            end: "2025-07-01T00:00:00.000Z",
          },
        });
        expect(results).toHaveLength(1);
      });

      it("returns all units when filter is empty", async () => {
        await storage.saveUnits([makeUnit(), makeUnit()]);
        const results = await storage.queryUnits({});
        expect(results).toHaveLength(2);
      });

      it("combines multiple filters (AND logic)", async () => {
        await storage.saveUnits([
          makeUnit({ persons: ["Alice"], location: "Starbucks" }),
          makeUnit({ persons: ["Alice"], location: "Office" }),
          makeUnit({ persons: ["Bob"], location: "Starbucks" }),
        ]);
        const results = await storage.queryUnits({
          persons: ["Alice"],
          location: "starbucks",
        });
        expect(results).toHaveLength(1);
      });
    });

    // --- Abstract Memories ---

    describe("abstract memories", () => {
      it("saves and retrieves abstracts", async () => {
        const abs = makeAbstract();
        await storage.saveAbstract(abs);
        const all = await storage.getAllAbstracts();
        expect(all).toHaveLength(1);
        expect(all[0].pattern).toBe(abs.pattern);
      });
    });

    // --- Clear ---

    describe("clear", () => {
      it("removes all units and abstracts", async () => {
        await storage.saveUnits([makeUnit(), makeUnit()]);
        await storage.saveAbstract(makeAbstract());
        await storage.clear();
        expect(await storage.getAllUnits()).toHaveLength(0);
        expect(await storage.getAllAbstracts()).toHaveLength(0);
      });
    });

    // --- Export / Import ---

    describe("export/import", () => {
      it("exports and re-imports data", async () => {
        const units = [makeUnit(), makeUnit()];
        const abs = makeAbstract();
        await storage.saveUnits(units);
        await storage.saveAbstract(abs);

        const exported = await storage.export();
        expect(exported.units).toHaveLength(2);
        expect(exported.abstracts).toHaveLength(1);
        expect(exported.version).toBe("1.0.0");
        expect(exported.exportedAt).toBeDefined();

        // Import into fresh storage
        const fresh = createStorage();
        await fresh.import(exported);
        expect(await fresh.getAllUnits()).toHaveLength(2);
        expect(await fresh.getAllAbstracts()).toHaveLength(1);
      });
    });
  });
}

// =============================================================================
// Run shared tests for each adapter
// =============================================================================

testStorageAdapter("MemoryStorage", () => new MemoryStorage());

const tmpFile = join(tmpdir(), `simplemem-test-${Date.now()}.json`);
testStorageAdapter(
  "FileStorage",
  () => new FileStorage({ path: tmpFile }),
  async () => {
    try {
      await rm(tmpFile, { force: true });
    } catch {}
  },
);

// =============================================================================
// FileStorage-specific tests
// =============================================================================

describe("FileStorage-specific", () => {
  const filePath = join(tmpdir(), `simplemem-fs-test-${Date.now()}.json`);

  afterEach(async () => {
    try {
      await rm(filePath, { force: true });
    } catch {}
  });

  it("persists data to disk and reloads", async () => {
    const fs1 = new FileStorage({ path: filePath });
    const unit = makeUnit();
    await fs1.saveUnit(unit);

    // Create new instance pointing to same file
    const fs2 = new FileStorage({ path: filePath });
    const retrieved = await fs2.getUnit(unit.id);
    expect(retrieved).not.toBeNull();
    expect(retrieved!.content).toBe(unit.content);
  });

  it("starts empty when file does not exist", async () => {
    const fs = new FileStorage({
      path: join(tmpdir(), `nonexistent-${Date.now()}.json`),
    });
    const all = await fs.getAllUnits();
    expect(all).toHaveLength(0);
  });

  it("respects autoSave=false", async () => {
    const fs1 = new FileStorage({ path: filePath, autoSave: false });
    await fs1.saveUnit(makeUnit());

    // Without save(), data shouldn't be on disk
    const fs2 = new FileStorage({ path: filePath });
    const all = await fs2.getAllUnits();
    expect(all).toHaveLength(0);
  });

  it("save() forces write when autoSave is disabled", async () => {
    const fs1 = new FileStorage({ path: filePath, autoSave: false });
    await fs1.saveUnit(makeUnit());
    await fs1.save();

    const fs2 = new FileStorage({ path: filePath });
    const all = await fs2.getAllUnits();
    expect(all).toHaveLength(1);
  });
});
