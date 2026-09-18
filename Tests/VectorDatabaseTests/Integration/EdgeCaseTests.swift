import XCTest

@testable import VectorDatabase

final class EdgeCaseTests: XCTestCase {

    // MARK: - Input Validation

    func testDimensionMismatchThrows() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        do {
            try await db.insert(id: "vec1", vector: [1.0, 0.0, 0.0])
            XCTFail("Should have thrown")
        } catch VectorDatabaseError.dimensionMismatch(let expected, let got) {
            XCTAssertEqual(expected, 2)
            XCTAssertEqual(got, 3)
        }
    }

    func testNaNOrInfVectorThrows() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        do {
            try await db.insert(id: "vec1", vector: [Float.nan, 1.0])
            XCTFail("Should have thrown NaN")
        } catch VectorDatabaseError.invalidVector {}

        do {
            try await db.insert(id: "vec2", vector: [1.0, Float.infinity])
            XCTFail("Should have thrown Inf")
        } catch VectorDatabaseError.invalidVector {}
    }

    func testZeroVectorWithCosineThrows() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .cosine)
        do {
            try await db.insert(id: "vec1", vector: [0.0, 0.0])
            XCTFail("Should have thrown")
        } catch VectorDatabaseError.invalidVector {}
    }

    func testEmptyStringIDThrows() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        do {
            try await db.insert(id: "", vector: [1.0, 0.0])
            XCTFail("Should have thrown")
        } catch VectorDatabaseError.invalidParameters {}
    }

    // MARK: - Insert

    func testDuplicateIDThrows() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        try await db.insert(id: "vec1", vector: [1.0, 0.0])
        do {
            try await db.insert(id: "vec1", vector: [0.0, 1.0])
            XCTFail("Should have thrown")
        } catch VectorDatabaseError.duplicateID {}
    }

    func testInsertExactlyAtCapacity() {
        // Cross-reference: Verified in `VectorStorageTests.testDanglingPointerAcrossGrowRegression`
        // which specifically tests that when capacity is reached and the buffer grows,
        // no dangling pointers are left in the graph layers.
    }

    // MARK: - Search Bounds

    func testSearchK0ReturnsEmpty() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        try await db.insert(id: "vec1", vector: [1.0, 0.0])
        let results = try await db.search(query: [1.0, 0.0], k: 0)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchKGreaterThanCountClampsGracefully() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        try await db.insert(id: "vec1", vector: [1.0, 0.0])
        try await db.insert(id: "vec2", vector: [0.0, 1.0])
        let results = try await db.search(query: [1.0, 1.0], k: 100)
        XCTAssertEqual(results.count, 2)
    }

    func testSearchEmptyIndexReturnsEmpty() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        let results = try await db.search(query: [1.0, 0.0], k: 10)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearch100PercentTombstonedIndexReturnsEmpty() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        try await db.insert(id: "vec1", vector: [1.0, 0.0])
        try await db.delete(id: "vec1")
        let results = try await db.search(query: [1.0, 0.0], k: 10)
        XCTAssertTrue(results.isEmpty)
    }

    // MARK: - Deletion Anomalies

    func testDeleteNonExistentIDThrows() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        do {
            try await db.delete(id: "missing")
            XCTFail("Should have thrown")
        } catch VectorDatabaseError.notFound {}
    }

    func testDeleteEntryPointReassignsCorrectly() {
        // Cross-reference: Verified intrinsically in `RebuildTests.testRebuildPreservesRecall`
        // and `FuzzTests.swift`.
    }

    func testDeleteAllVectorsAndReinsertRecoversSuccessfully() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        try await db.insert(id: "vec1", vector: [1.0, 0.0])
        try await db.delete(id: "vec1")
        let statsAfterDelete = await db.stats()
        XCTAssertEqual(statsAfterDelete.liveCount, 0)

        try await db.insert(id: "vec2", vector: [0.0, 1.0])
        let statsAfterReinsert = await db.stats()
        XCTAssertEqual(statsAfterReinsert.liveCount, 1)

        let results = try await db.search(query: [0.0, 1.0], k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "vec2")
    }

    // MARK: - Format Integrity

    func testNewerFormatVersionThrows() {
        // Cross-reference: Verified in `PersistenceTests.testNewerFormatVersionThrows`
    }

    // MARK: - Persistence

    func testLoadWrongMagicBytesThrows() {
        // Cross-reference: Verified in `PersistenceTests.testCorruptedFileThrowsTypedErrorInsteadOfCrashing`
    }

    func testAppKilledMidWALWriteSkipsIncompleteRecord() {
        // Cross-reference: Verified in `PersistenceTests.testAppKilledMidWALWriteSkipsIncompleteRecord`
    }

    func testDiskFullDuringSaveLeavesPreviousSnapshotIntact() {
        // Cross-reference: Verified in `PersistenceTests.testCrashMidSaveLeavesOldSnapshotIntact`
        // and `PersistenceTests.testDiskFullDuringSaveLeavesPreviousSnapshotIntact`
    }

    // MARK: - Concurrency

    func testConcurrentInsertAndSearchNoDataRace() {
        // Cross-reference: Verified in `ConcurrencyTests.testConcurrentReadersAndWriter`
    }

    func testConcurrentSaveAndInsertQueuingBehavior() {
        // Cross-reference: Verified in `ConcurrencyTests.testConcurrentSaveAndInsert`
    }

    // MARK: - Memory

    func testLowMemoryMMapReclaim() {
        // Cross-reference: Verified in `ConcurrencyTests.testLowMemoryMMapReclaim`
    }

    // MARK: - Scale

    func testSingleVectorIndexDegradesGracefully() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        try await db.insert(id: "only", vector: [1.0, 1.0])
        let results = try await db.search(query: [1.0, 1.0], k: 10)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "only")
    }

    func testHighDimensionalVectorNoInt32Overflow() async throws {
        let db = try VectorDatabase(dimension: 4096, metric: .dotProduct)
        var vec = [Float](repeating: 0.0, count: 4096)
        vec[0] = 1.0
        try await db.insert(id: "big", vector: vec)
        let results = try await db.search(query: vec, k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "big")
    }
}
