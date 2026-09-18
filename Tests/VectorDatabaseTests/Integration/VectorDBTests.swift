import XCTest

@testable import VectorDatabase

final class VectorDatabaseTests: XCTestCase {

    var dbURL: URL!

    override func setUp() {
        super.setUp()
        let tempDir = FileManager.default.temporaryDirectory
        dbURL = tempDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("vdb")
    }

    override func tearDown() {
        if FileManager.default.fileExists(atPath: dbURL.path) {
            try? FileManager.default.removeItem(at: dbURL)
        }
        super.tearDown()
    }

    func testInMemoryModeAndDuplicateIDPolicy() async throws {
        // Initialize in-memory (no path)
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)

        // 1. Insert
        try await db.insert(id: "vec1", vector: [1.0, 0.0], metadata: ["type": "test"])

        let stats1 = await db.stats()
        XCTAssertEqual(stats1.liveCount, 1)

        // 2. Duplicate ID must throw
        do {
            try await db.insert(id: "vec1", vector: [0.0, 1.0])
            XCTFail("Should throw .duplicateID")
        } catch VectorDatabaseError.duplicateID(let id) {
            XCTAssertEqual(id, "vec1")
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }

    func testUpdateIsExplicitUpsert() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)

        try await db.insert(id: "vec1", vector: [1.0, 0.0])

        // Update to new vector and new metadata
        try await db.update(id: "vec1", vector: [0.0, 1.0], metadata: ["updated": "true"])

        let results = try await db.search(query: [0.0, 1.0], k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "vec1")
        XCTAssertEqual(results[0].metadata?["updated"], "true")

        // FlatIndex uses swap-remove, so there are no tombstones (tombstonedCount is 0).
        // If we crossed the hnswThreshold (2000), it would be 1.
        let stats = await db.stats()
        XCTAssertEqual(stats.liveCount, 1)
    }

    func testBatchInsertAndSearch() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .cosine)

        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            (id: "doc1", vector: [1.0, 0.0], metadata: ["topic": "A"]),
            (id: "doc2", vector: [0.0, 1.0], metadata: ["topic": "B"]),
            (id: "doc3", vector: [-1.0, 0.0], metadata: ["topic": "C"]),
        ]

        try await db.batchInsert(batch)

        let stats = await db.stats()
        XCTAssertEqual(stats.liveCount, 3)

        let results = try await db.search(query: [1.0, 0.0], k: 2)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].id, "doc1")  // Exactly matches [1.0, 0.0]
        XCTAssertEqual(results[0].metadata?["topic"], "A")
    }

    func testPersistenceRoundTrip() async throws {
        // Create, insert, save
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        try await db.insert(id: "a", vector: [1.0, 2.0], metadata: ["foo": "bar"])
        try await db.save()
        await db.close()

        // Load in a fresh instance
        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        let stats = await db2.stats()

        XCTAssertEqual(stats.liveCount, 1)
        let results = try await db2.search(query: [1.0, 2.0], k: 1)
        XCTAssertEqual(results.first?.id, "a")
        XCTAssertEqual(results.first?.metadata?["foo"], "bar")
    }

    func testCloseIsIdempotent() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        await db.close()
        await db.close()  // Should not crash

        do {
            try await db.insert(id: "vec1", vector: [1.0, 1.0])
            XCTFail("Operations on closed DB should fail")
        } catch VectorDatabaseError.invalidParameters {
            // Expected
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }

    func testEnergyProfileHNSWIOSDevice() async throws {
        // This test is specifically designed to be profiled via Instruments (Energy Log)
        // on a physical iOS device. It runs a sustained workload of inserts and searches.
        // It is skipped by default to avoid slowing down CI/local test runs.
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["RUN_ENERGY_PROFILE"] != nil,
            "Skipping energy profile. Run via Xcode Instruments on device.")

        let dim = 128
        let db = try VectorDatabase(dimension: dim, metric: .cosine)

        var rng = SeedableRNG(seed: 0x5EED)

        print("Starting Energy Profiling Workload for 15 seconds...")
        let start = Date()
        var insertCount = 0
        var searchCount = 0

        while Date().timeIntervalSince(start) < 15.0 {
            // Interleave inserts and searches to simulate real workload
            for i in 0..<100 {
                let vec = (0..<dim).map { _ in rng.nextFloat() }
                try await db.insert(id: "doc_\(insertCount + i)", vector: vec)
            }
            insertCount += 100

            for _ in 0..<50 {
                let query = (0..<dim).map { _ in rng.nextFloat() }
                _ = try await db.search(query: query, k: 5)
                searchCount += 1
            }
        }

        print("Completed Energy Profiling: \(insertCount) inserts, \(searchCount) searches.")
    }
}
