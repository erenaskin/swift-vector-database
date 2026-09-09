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
}
