import XCTest

@testable import VectorDatabase

final class DeletionTests: XCTestCase {

    // 1. Delete 30% of a 20k-vector index, verify search never returns a tombstoned ID.
    func testDelete30PercentAndSearch() throws {
        let dim = 16
        let count = 20_000
        let engine = Engine(dimension: dim, metric: .dotProduct, hnswThreshold: 1)  // Force HNSW

        // Insert 20k
        for i in 0..<Int32(count) {
            let vec = (0..<dim).map { _ in Float.random(in: -1...1) }
            try vec.withUnsafeBufferPointer { buf in
                try engine.insert(internalID: i, vector: buf.baseAddress!)
            }
        }

        print("--- Inserted 20k ---")
        XCTAssertTrue(engine.stats.isUsingHNSW)
        XCTAssertEqual(engine.stats.liveCount, count)

        // Delete 30% (6,000 vectors)
        let deleteCount = Int(Double(count) * 0.3)
        let deletedIDs = Set(
            (0..<Int32(deleteCount)).map { _ in Int32.random(in: 0..<Int32(count)) })

        print("--- Deleting \(deletedIDs.count) vectors ---")
        var countDeleted = 0
        for id in deletedIDs {
            try? engine.remove(internalID: id)
            countDeleted += 1
            if countDeleted % 1000 == 0 { print("Deleted \(countDeleted)") }
        }

        print("--- Verifying tombstone stats ---")

        XCTAssertEqual(engine.stats.tombstonedCount, deletedIDs.count)

        print("--- Searching ---")
        // Verify searches never return tombstoned IDs
        for i in 0..<100 {
            if i % 25 == 0 { print("Search \(i)") }
            let query = (0..<dim).map { _ in Float.random(in: -1...1) }
            let results = query.withUnsafeBufferPointer { buf in
                engine.search(query: buf.baseAddress!, k: 10, ef: 50)
            }
            for (id, _) in results {
                XCTAssertFalse(deletedIDs.contains(id), "Search returned a tombstoned ID: \(id)")
            }
        }
        print("--- Done ---")
    }

    // 3. Specifically tombstone the current entry point and verify subsequent inserts/searches succeed.
    func testEntryPointTombstoning() throws {
        let dim = 4
        let engine = Engine(dimension: dim, metric: .dotProduct, hnswThreshold: 1)

        // Insert node 0
        let vec0 = [Float](repeating: 1.0, count: dim)
        try vec0.withUnsafeBufferPointer { buf in
            try engine.insert(internalID: 0, vector: buf.baseAddress!)
        }

        // Insert node 1
        let vec1 = [Float](repeating: -1.0, count: dim)
        try vec1.withUnsafeBufferPointer { buf in
            try engine.insert(internalID: 1, vector: buf.baseAddress!)
        }

        // Tombstone node 0 (the original entry point)
        try engine.remove(internalID: 0)

        // Insert node 2
        let vec2 = [Float](repeating: 0.5, count: dim)
        try vec2.withUnsafeBufferPointer { buf in
            try engine.insert(internalID: 2, vector: buf.baseAddress!)
        }

        // Search should work and not crash, and shouldn't return node 0
        let query = [Float](repeating: 1.0, count: dim)
        let results = query.withUnsafeBufferPointer { buf in
            engine.search(query: buf.baseAddress!, k: 10)
        }

        let ids = results.map(\.id)
        XCTAssertFalse(ids.contains(0), "Tombstoned entry point was returned in search!")
        XCTAssertTrue(ids.contains(1))
        XCTAssertTrue(ids.contains(2))
    }

    /// The tests above delete a subset (30%, or a single entry point) and always
    /// leave at least one living node behind. This test covers the edge case none
    /// of them reach: deleting EVERY vector, leaving the graph fully empty, and
    /// then inserting again. `HNSWIndex.remove` explicitly resets `entryPoint` to
    /// `nil` when the last living node is removed (see the fallback branch in
    /// `remove()`), and `insert()` explicitly re-establishes a fresh entry point
    /// when `entryPoint` is `nil` — but nothing exercised that specific transition
    /// end-to-end before.
    func testDeleteAllThenReinsertReestablishesEntryPoint() throws {
        let dim = 4
        let engine = Engine(dimension: dim, metric: .dotProduct, hnswThreshold: 1)  // Force HNSW

        let vectors: [Int32: [Float]] = [
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
        ]
        for (id, vec) in vectors {
            try vec.withUnsafeBufferPointer { buf in
                try engine.insert(internalID: id, vector: buf.baseAddress!)
            }
        }
        XCTAssertEqual(engine.stats.liveCount, 3)

        // Delete every single vector — the graph should end up completely empty,
        // not just tombstoned-but-nonzero.
        for id in vectors.keys {
            try engine.remove(internalID: id)
        }
        XCTAssertEqual(engine.stats.liveCount, 0)

        // Searching a fully empty graph must return no results, not crash.
        let probeQuery: [Float] = [1, 0, 0, 0]
        let emptyResults = probeQuery.withUnsafeBufferPointer { buf in
            engine.search(query: buf.baseAddress!, k: 5, ef: 20)
        }
        XCTAssertTrue(emptyResults.isEmpty, "Searching a fully empty graph should return no results")

        // Inserting into the now-empty graph must re-establish a working entry
        // point from scratch, not silently fail or reuse a stale/nil reference.
        let freshVector: [Float] = [0, 0, 0, 1]
        try freshVector.withUnsafeBufferPointer { buf in
            try engine.insert(internalID: 99, vector: buf.baseAddress!)
        }
        XCTAssertEqual(engine.stats.liveCount, 1)

        let resultsAfterReinsert = freshVector.withUnsafeBufferPointer { buf in
            engine.search(query: buf.baseAddress!, k: 1, ef: 20)
        }
        XCTAssertEqual(resultsAfterReinsert.first?.id, 99,
            "A fresh insert after fully emptying the graph must be searchable, proving the "
            + "entry point was correctly re-established rather than left dangling")
    }
    
    // MARK: - [DoD-F] remove() entry point fallback (PROMPT 2)
    
    func testRemoveEntryPointWithLivingNeighborsAtSameLevel() throws {
        // Fast path: the entry point is removed, but it has a living neighbor at its level.
        // We use a fixed seed and known vectors to guarantee node 0 is entry point and node 1 connects to it at the same top level.
        let params = HNSWParameters(M: 4, efConstruction: 16, efSearch: 8, seed: 42)
        var index = HNSWIndex(dimension: 2, metric: .dotProduct, params: params)
        // Force nodes to go to level 2
        index.rng = SeedableRNG(seed: 42) // just to be deterministic
        
        // Let's explicitly build an index where node 0 is at level 2, node 1 is at level 2.
        // To do this reliably, we can just insert them and override their levels or just use a mock,
        // but since `HNSWIndex` relies on `rng`, we can just override `randomLevel` by manually calling insert?
        // Wait, `nodes` is internal, we can just mutate `nodes` and `entryPointLevel` directly.
        index.entryPoint = 0
        index.entryPointLevel = 2
        
        index.setNode(HNSWNode(level: 2, vectorSlot: 0), for: 0)
        index.setNode(HNSWNode(level: 2, vectorSlot: 1), for: 1)
        index.graphStorage.addNode() // node 0
        index.graphStorage.addNode() // node 1
        
        index.graphStorage.addNeighbor(of: 0, at: 2, neighborID: 1)
        index.graphStorage.addNeighbor(of: 1, at: 2, neighborID: 0)
        
        // Remove 0
        try index.remove(internalID: 0)
        
        // The new entry point should be 1, and level should remain 2.
        XCTAssertEqual(index.entryPoint, 1)
        XCTAssertEqual(index.entryPointLevel, 2)
    }
    
    func testRemoveEntryPointWithNoLivingNeighborsAtSameLevel() throws {
        // Fallback path: the entry point is removed, and it has NO living neighbors at its level.
        // It must fallback to O(N) scan.
        let params = HNSWParameters(M: 4, efConstruction: 16, efSearch: 8, seed: 42)
        var index = HNSWIndex(dimension: 2, metric: .dotProduct, params: params)
        
        index.entryPoint = 0
        index.entryPointLevel = 3
        
        index.setNode(HNSWNode(level: 3, vectorSlot: 0), for: 0)
        index.setNode(HNSWNode(level: 1, vectorSlot: 1), for: 1) // Next highest is level 1
        index.setNode(HNSWNode(level: 2, vectorSlot: 2), for: 2) // Wait, this one is level 2, but it's tombstoned
        
        index.tombstoned.insert(2)
        
        // Remove 0
        try index.remove(internalID: 0)
        
        // Since 0 has no neighbors at level 3 (none added), it falls back.
        // The highest surviving is node 1 at level 1 (since node 2 is tombstoned).
        XCTAssertEqual(index.entryPoint, 1)
        XCTAssertEqual(index.entryPointLevel, 1)
    }
}
