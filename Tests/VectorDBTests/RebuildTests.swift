import XCTest

@testable import VectorDB

final class RebuildTests: XCTestCase {

    // 2. Trigger a rebuild and verify recall@10 against FlatIndex recovers to baseline.
    func testRebuildPreservesRecall() throws {
        let dim = 16
        let count = 5_000
        print(">> testRebuildPreservesRecall: start")
        fflush(stdout)
        let engineHNSW = Engine(dimension: dim, metric: .cosine, hnswThreshold: 1)
        var flatIndex = FlatIndex(dimension: dim, metric: .cosine)

        print(">> testRebuildPreservesRecall: inserting \(count) vectors")
        fflush(stdout)
        var vectors: [[Float]] = []
        for i in 0..<Int32(count) {
            let vec = (0..<dim).map { _ in Float.random(in: -1...1) }
            vectors.append(vec)
            try vec.withUnsafeBufferPointer { buf in
                try engineHNSW.insert(internalID: i, vector: buf.baseAddress!)
                try flatIndex.insert(internalID: i, vector: buf.baseAddress!)
            }
        }

        print(">> testRebuildPreservesRecall: tombstoning")
        fflush(stdout)
        // Tombstone 20% (exactly at the >= threshold — previously caused off-by-one)
        let deleteCount = count / 5  // = 1_000 = exactly 20%
        for i in 0..<Int32(deleteCount) {
            try engineHNSW.remove(internalID: i)
            try flatIndex.remove(internalID: i)
        }
        print(
            ">> testRebuildPreservesRecall: tombstoned, shouldRebuild=\(engineHNSW.shouldRebuild)")
        fflush(stdout)

        // Exactly at the threshold boundary: 1000/5000 = 20% >= 0.20 → should trigger
        XCTAssertTrue(engineHNSW.shouldRebuild, "Rebuild should trigger at exactly 20% tombstones")

        print(">> testRebuildPreservesRecall: collectLiveSnapshots")
        fflush(stdout)
        // Collect an owned live-vector snapshot via the encapsulated Engine method.
        // Copies vectors into [Float] before the write lock is acquired for rebuild,
        // preventing dangling pointers if VectorStorage remaps during the rebuild.
        let liveSnapshots = engineHNSW.collectLiveSnapshots()
        XCTAssertEqual(liveSnapshots.count, count - deleteCount)

        print(">> testRebuildPreservesRecall: rebuild")
        fflush(stdout)
        try engineHNSW.rebuild(with: liveSnapshots)
        print(">> testRebuildPreservesRecall: rebuild done")
        fflush(stdout)

        XCTAssertFalse(
            engineHNSW.shouldRebuild, "shouldRebuild must be false after a clean rebuild")
        XCTAssertEqual(engineHNSW.stats.tombstonedCount, 0)
        XCTAssertEqual(engineHNSW.stats.liveCount, count - deleteCount)

        // Measure Recall@10 against the FlatIndex oracle
        var exactMatches = 0
        let queryCount = 100
        print(">> testRebuildPreservesRecall: query")
        fflush(stdout)

        for _ in 0..<queryCount {
            let query = (0..<dim).map { _ in Float.random(in: -1...1) }

            let hnswResults = query.withUnsafeBufferPointer { buf in
                engineHNSW.search(query: buf.baseAddress!, k: 10, ef: 50)
            }

            let flatResults = query.withUnsafeBufferPointer { buf in
                flatIndex.search(query: buf.baseAddress!, k: 10)
            }

            let hnswIDs = Set(hnswResults.map(\.id))
            let flatIDs = Set(flatResults.map(\.id))
            exactMatches += hnswIDs.intersection(flatIDs).count
        }

        let recall = Double(exactMatches) / Double(queryCount * 10)
        print("Recall after rebuild: \(recall)")
        XCTAssertGreaterThan(recall, 0.95, "Recall dropped below 0.95 after rebuild!")
    }
    
    // MARK: - Task 4: VectorDB Actor Level Tests
    
    func testAutomaticRebuildOnDelete() async throws {
        // hnswThreshold is 2000. Insert 2005 to trigger HNSW mode.
        let db = try VectorDB(dimension: 2, metric: .dotProduct, parameters: HNSWParameters(M: 16, efConstruction: 100))
        for i in 1...2005 {
            try await db.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        
        // Threshold is 20% of 2005 = 401. 
        // Delete 400. 400 < 20%
        for i in 1...400 {
            try await db.delete(id: "v\(i)")
        }
        var stats = await db.stats()
        XCTAssertEqual(stats.tombstonedCount, 400)
        
        // Delete 1 more. 401/2005 >= 20%, so automatic rebuild triggers
        try await db.delete(id: "v401")
        stats = await db.stats()
        
        // After rebuild, tombstonedCount should be 0
        XCTAssertEqual(stats.tombstonedCount, 0, "Automatic rebuild should have triggered and cleared tombstones")
        XCTAssertEqual(stats.liveCount, 2005 - 401)
    }
    
    func testAutomaticRebuildOnUpdate() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct, parameters: HNSWParameters(M: 16, efConstruction: 100))
        for i in 1...2005 {
            try await db.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        
        // update adds 1 tombstone but keeps liveCount at 2005
        // ratio = tombstones / (liveCount + tombstones)
        // 20% threshold: T / (2005 + T) = 0.2 => T = 501.25
        // So 501 updates = 501 / 2506 = 19.99% < 20%
        for i in 1...501 {
            try await db.update(id: "v\(i)", vector: [1.1, 1.1])
        }
        var stats = await db.stats()
        XCTAssertEqual(stats.tombstonedCount, 501)
        
        // 502 updates = 502 / 2507 = 20.02% >= 20% -> triggers rebuild!
        try await db.update(id: "v502", vector: [2.2, 2.2])
        stats = await db.stats()
        
        XCTAssertEqual(stats.tombstonedCount, 0, "Automatic rebuild should have triggered and cleared tombstones")
        XCTAssertEqual(stats.liveCount, 2005)
    }
    
    func testManualCompactNoOp() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct, parameters: HNSWParameters(M: 16, efConstruction: 100))
        for i in 1...2005 {
            try await db.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        for i in 1...100 {
            try await db.delete(id: "v\(i)")
        }
        
        let stats = await db.stats()
        XCTAssertEqual(stats.tombstonedCount, 100) // ~5%
        
        let compacted = try await db.compact()
        XCTAssertFalse(compacted, "Should return false if ratio is below threshold")
        
        let postStats = await db.stats()
        XCTAssertEqual(postStats.tombstonedCount, 100, "Should not rebuild if ratio is below threshold")
    }
    
    func testManualCompactTriggersRebuild() async throws {
        // We need to bypass the automatic trigger to test the manual trigger.
        // Wait, if automatic trigger fires on delete, how can we have a DB above threshold without triggering it?
        // Let's insert 5. Threshold is 20%.
        // Wait, the automatic trigger ALWAYS keeps the database below the threshold!
        // To test the manual trigger, we can lower the threshold via Engine? 
        // Wait, Engine.shouldRebuild uses a hardcoded 0.20 threshold.
        // If automatic rebuild is on, `compact()` will NEVER return true in normal usage unless an insert/update somehow fails halfway or we do batch deletes? Wait, delete calls it.
        // Is there any way to get the tombstone ratio to 20% without going through delete/update?
        // No, because idMap only lets you delete via delete/update.
        // Wait... test testManualCompactTriggersRebuild is explicitly requested: "A test that manually calls compact() right after crossing the threshold (simulating a developer who wants to force it) and asserts it returns true."
        // Wait, if it crosses the threshold, the automatic trigger fires first!
        // Is it possible the spec author missed this logical contradiction? "A test that manually calls compact() right after crossing the threshold ... and asserts it returns true."
        // Since delete/update trigger it automatically, compact() will return false.
        // Wait, what if we use reflection to set engine's tombstone count, or bypass VectorDB?
        // Let's use Mirror to extract `engine` and insert/remove directly, then call `db.compact()`!
        
        let db = try VectorDB(dimension: 2, metric: .dotProduct)
        for i in 1...2005 {
            try await db.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        
        let mirror = Mirror(reflecting: db)
        guard let engine = mirror.children.first(where: { $0.label == "engine" })?.value as? Engine else {
            XCTFail()
            return
        }
        
        // Remove bypassing VectorDB auto-trigger
        for i in 1...401 {
            // Note: internalID maps 1-to-1 with insertions initially
            try engine.remove(internalID: Int32(i - 1)) // 20%
        }
        
        let compacted = try await db.compact()
        XCTAssertTrue(compacted, "Should return true since threshold was met")
        
        let stats = await db.stats()
        XCTAssertEqual(stats.tombstonedCount, 0)
    }
}
