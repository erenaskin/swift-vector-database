import XCTest

@testable import VectorDatabase

final class CosineNormalizationTests: XCTestCase {

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

    func testCosineNormalizationOnInsertAndSearch() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .cosine, path: nil)

        let vec1: [Float] = [3.0, 4.0]  // magnitude 5
        let vec2: [Float] = [6.0, 8.0]  // magnitude 10

        try await db.insert(id: "v1", vector: vec1)
        try await db.insert(id: "v2", vector: vec2)

        // Search with vec1
        let results = try await db.search(query: vec1, k: 2)
        XCTAssertEqual(results.count, 2)

        for result in results {
            // For unit vectors, dot product = 1.0 (cosine of 0)
            XCTAssertEqual(result.score, 1.0, accuracy: 1e-4)
        }
    }

    func testCosineNormalizationConsistencyOnUpdate() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .cosine, path: nil)

        let vec1: [Float] = [3.0, 4.0]
        try await db.insert(id: "v1", vector: vec1)

        let vec2: [Float] = [6.0, 8.0]
        try await db.insert(id: "v2", vector: [1.0, 0.0])  // dummy
        try await db.update(id: "v2", vector: vec2)

        // Both should be treated consistently as normalized
        let results = try await db.search(query: vec1, k: 2)
        XCTAssertEqual(results.count, 2)
        for result in results {
            XCTAssertEqual(result.score, 1.0, accuracy: 1e-4)
        }
    }

    func testWALRecoveryPreservesNormalizedVectors() async throws {
        let vec1: [Float] = [3.0, 4.0]
        let vec2: [Float] = [6.0, 8.0]

        var liveResults: [SearchResult] = []

        // 1. Live index
        do {
            let db = try VectorDatabase(dimension: 2, metric: .cosine, path: dbURL)
            try await db.save()  // Create base file so recovery works
            try await db.insert(id: "v1", vector: vec1)
            try await db.insert(id: "v2", vector: vec2)

            liveResults = try await db.search(query: vec1, k: 2)

            // Do NOT call save() again, to force WAL replay
            // But we must flush the WAL to disk so the next instance can read it,
            // since VectorDatabase.insert doesn't fsync yet (Task 3).
            let mirror = Mirror(reflecting: db)
            if let pm = mirror.children.first(where: { $0.label == "persistenceManager" })?.value
                as? PersistenceManager
            {
                try pm.wal?.fsync()
            }
        }

        // 2. Recovered index
        let recoveredDB = try VectorDatabase(dimension: 2, metric: .cosine, path: dbURL)

        let recoveredMirror = Mirror(reflecting: recoveredDB)
        guard
            let recoveredEngine = recoveredMirror.children.first(where: { $0.label == "engine" })?
                .value as? AnyObject
        else {
            XCTFail("Could not access engine")
            return
        }

        // Use perform method or cast. Engine is internal but we have @testable import VectorDatabase
        // So we can just cast to Engine!
        guard let engine = recoveredEngine as? Engine else {
            XCTFail("Could not cast to Engine")
            return
        }

        var finalVec1 = vec1
        let recoveredResults = finalVec1.withUnsafeMutableBufferPointer {
            buf -> [(id: Int32, score: Float)] in
            guard let ptr = buf.baseAddress else { return [] }
            VectorMath.normalize(ptr, buf.count)  // Query must be normalized for Engine!
            return engine.search(query: ptr, k: 2, ef: nil)
        }

        XCTAssertEqual(liveResults.count, 2)
        XCTAssertEqual(recoveredResults.count, 2)

        for (live, recovered) in zip(
            liveResults.sorted(by: { $0.score > $1.score }),
            recoveredResults.sorted(by: { $0.score > $1.score }))
        {
            XCTAssertEqual(live.score, recovered.score, accuracy: 1e-4)
            XCTAssertEqual(recovered.score, 1.0, accuracy: 1e-4)
        }
    }

    func testNearZeroVectorThrowsInvalidVector() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .cosine, path: nil)

        // Vector length is 2. Elements are 1e-10. SumSq = 2e-20
        // This is below the 1e-12 threshold.
        let nearZeroVec: [Float] = [1e-10, 1e-10]

        do {
            try await db.insert(id: "near_zero", vector: nearZeroVec)
            XCTFail("Expected VectorDatabaseError.invalidVector to be thrown")
        } catch VectorDatabaseError.invalidVector {
            // Success
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testDotProductAndEuclideanMetricsNotNormalized() async throws {
        // DotProduct
        let dbDot = try VectorDatabase(dimension: 2, metric: .dotProduct, path: nil)
        let vec1: [Float] = [3.0, 4.0]
        let vec2: [Float] = [6.0, 8.0]
        try await dbDot.insert(id: "v1", vector: vec1)
        try await dbDot.insert(id: "v2", vector: vec2)
        let resultsDot = try await dbDot.search(query: vec1, k: 2)

        // DotProduct [3,4] . [3,4] = 25
        // DotProduct [3,4] . [6,8] = 50
        if let v1 = resultsDot.first(where: { $0.id == "v1" }) {
            XCTAssertEqual(v1.score, 25.0, accuracy: 1e-4)
        } else {
            XCTFail("v1 not found")
        }
        if let v2 = resultsDot.first(where: { $0.id == "v2" }) {
            XCTAssertEqual(v2.score, 50.0, accuracy: 1e-4)
        } else {
            XCTFail("v2 not found")
        }

        // Euclidean
        let dbEuclid = try VectorDatabase(dimension: 2, metric: .euclidean, path: nil)
        try await dbEuclid.insert(id: "v1", vector: vec1)
        try await dbEuclid.insert(id: "v2", vector: vec2)
        let resultsEuclid = try await dbEuclid.search(query: vec1, k: 2)

        // Euclidean [3,4] to [3,4] = -0
        // Euclidean [3,4] to [6,8] = -(3^2 + 4^2) = -25
        if let v1 = resultsEuclid.first(where: { $0.id == "v1" }) {
            XCTAssertEqual(v1.score, 0.0, accuracy: 1e-4)
        } else {
            XCTFail("v1 not found")
        }
        if let v2 = resultsEuclid.first(where: { $0.id == "v2" }) {
            XCTAssertEqual(v2.score, -25.0, accuracy: 1e-4)
        } else {
            XCTFail("v2 not found")
        }
    }
}
