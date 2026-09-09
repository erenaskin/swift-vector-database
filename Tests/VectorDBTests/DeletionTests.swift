import XCTest

@testable import VectorDB

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
}
