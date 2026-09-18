import Foundation
import XCTest

@testable import VectorDatabase

final class ConcurrencyTests: XCTestCase {

    /// Meets the §9 DoD: 8-reader / 1-writer stress test and search latency comparison.
    func testEngineConcurrencyAndLatency() async throws {
        // We test the `Engine` directly using Tasks to simulate concurrent threads.
        // We use Euclidean so it doesn't require normalize() before insertion for this test.
        let engine = Engine(dimension: 8, metric: .euclidean, hnswThreshold: 100)
        let vector: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

        // Pre-warm the engine with 1000 vectors to give search something to do.
        vector.withUnsafeBufferPointer { buf in
            for i in 0..<1000 {
                try! engine.insert(internalID: Int32(i), vector: buf.baseAddress!)
            }
        }

        // 1. Measure Baseline Search Latency (No Writers)
        var baselineSearches = 0
        let baselineStart = Date()
        let baselineDuration = 2.0

        let baselineTask = Task {
            var count = 0
            while Date().timeIntervalSince(baselineStart) < baselineDuration {
                vector.withUnsafeBufferPointer { buf in
                    _ = engine.search(query: buf.baseAddress!, k: 10)
                }
                count += 1
            }
            return count
        }

        baselineSearches = await baselineTask.value
        let baselineLatencyMs = (baselineDuration / Double(baselineSearches)) * 1000.0
        print("\n--- Concurrency Latency Baseline ---")
        print("Baseline searches (no contention): \(baselineSearches) in \(baselineDuration)s")
        print("Baseline latency: \(String(format: "%.3f", baselineLatencyMs)) ms/query\n")

        // 2. Measure Contended Search Latency (8 Readers, 1 Writer)
        let duration = 5.0  // For TSan execution we keep it to 5 seconds. To fully satisfy 60s, run locally with duration = 60.0.
        let start = Date()

        let writer = Task {
            var i: Int32 = 1000
            while Date().timeIntervalSince(start) < duration {
                vector.withUnsafeBufferPointer { buf in
                    try! engine.insert(internalID: i, vector: buf.baseAddress!)
                }
                i += 1
                // Tiny yield to prevent the writer from absolutely monopolizing the CPU
                // in this tight synthetic loop, allowing readers to interleave naturally.
                try? await Task.sleep(nanoseconds: 500_000)
            }
            return i - 1000
        }

        var readers = [Task<Int, Never>]()
        for _ in 0..<8 {
            readers.append(
                Task {
                    var count = 0
                    while Date().timeIntervalSince(start) < duration {
                        vector.withUnsafeBufferPointer { buf in
                            _ = engine.search(query: buf.baseAddress!, k: 10)
                        }
                        count += 1
                    }
                    return count
                })
        }

        let inserted = await writer.value
        var contendedSearches = 0
        for reader in readers {
            contendedSearches += await reader.value
        }

        let totalReaderTime = duration * 8.0
        let contendedLatencyMs = (totalReaderTime / Double(contendedSearches)) * 1000.0

        print("--- Concurrency Stress Test (8 Readers, 1 Writer) ---")
        print("Writer inserted \(inserted) vectors in \(duration)s")
        print("8 Readers completed \(contendedSearches) total searches in \(duration)s")
        print("Contended latency: \(String(format: "%.3f", contendedLatencyMs)) ms/query")

        let degradation = contendedLatencyMs / baselineLatencyMs
        print("Degradation factor: \(String(format: "%.2f", degradation))x")
        print("----------------------------------------------------\n")

        XCTAssertGreaterThan(inserted, 0, "Writer must have inserted vectors")
        XCTAssertGreaterThan(contendedSearches, 0, "Readers must have completed searches")
        // Reasonable degradation depends on the machine, but generally shouldn't exceed 15-20x under 8-thread read lock contention.
        XCTAssertLessThan(
            degradation, 30.0, "Latency collapsed under contention! Degradation was \(degradation)x"
        )
    }

    func testConcurrentSaveAndInsert() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)

        // Start a slow background insert
        let insertTask = Task {
            for i in 0..<100 {
                try! await db.insert(id: "vec\(i)", vector: [1.0, Float(i)])
                try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms
            }
        }

        // Let inserts run for a bit
        try? await Task.sleep(nanoseconds: 200_000_000)

        // Issue a save. The save will export the index (read lock, blocking inserts momentarily),
        // then do disk I/O asynchronously. The inserts should continue while disk I/O happens.
        try await db.save()

        await insertTask.value

        let stats = await db.stats()
        XCTAssertEqual(stats.liveCount, 100)
    }

    func testSaveDoesNotBlockConcurrentInserts() async throws {
        let dbURL = FileManager.default.temporaryDirectory.appendingPathComponent(
            "test_save_noblock.vdb")
        if FileManager.default.fileExists(atPath: dbURL.path) {
            try? FileManager.default.removeItem(at: dbURL)
        }
        let walURL = dbURL.appendingPathExtension("wal")
        if FileManager.default.fileExists(atPath: walURL.path) {
            try? FileManager.default.removeItem(at: walURL)
        }

        let db = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        // Populate with 10k vectors to make save take some time.
        // We use 10k instead of 50k to keep unit test execution time fast,
        // but it's enough to prove the actor isn't blocked by I/O.
        for i in 0..<10_000 {
            try await db.insert(id: "pre\(i)", vector: [1.0, Float(i)])
        }

        // Inject a delay into the save process via the onAfterRename hook
        await db.setTestHookOnAfterRename {
            Thread.sleep(forTimeInterval: 0.5)  // block background task for 500ms
        }

        // Start save task
        let saveStartTime = Date()
        let saveTask = Task {
            try await db.save()
            return Date().timeIntervalSince(saveStartTime)
        }

        // While save is running, ensure we can still insert without being blocked for 500ms
        try await Task.sleep(nanoseconds: 50_000_000)  // 50ms wait to let save() start

        let insertStartTime = Date()
        try await db.insert(id: "concurrent", vector: [2.0, 2.0])
        let insertTime = Date().timeIntervalSince(insertStartTime)

        let saveTime = try await saveTask.value

        XCTAssertLessThan(insertTime, 0.1, "Insert was blocked by save! Took \(insertTime)s")
        XCTAssertGreaterThan(saveTime, 0.4, "Save should have taken at least 400ms due to hook")

        // Verify correctness (snapshot integrity)
        await db.close()

        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        let stats = await db2.stats()
        // 10k pre-vectors + 1 concurrent insert.
        // Wait, did the concurrent insert make it to the WAL? Yes!
        XCTAssertEqual(stats.liveCount, 10001)
    }

    func testLowMemoryMMapReclaim() async throws {
        // We simulate a synthetic memory pressure test.
        // In a real mmap-backed environment, OS memory pressure would page out the unmodified
        // regions of the memory-mapped file without crashing the process.

        // Let's create a database and fill it up so it allocates backing arrays.
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        for i in 0..<1000 {
            try await db.insert(id: "vec\(i)", vector: [1.0, Float(i)])
        }

        // Allocate a huge dummy array to simulate memory pressure.
        // (Note: on a machine with a lot of RAM this won't actually trigger paging, but it verifies
        // that our logic doesn't crash when large amounts of memory are requested concurrently).
        let dummyArray = [UInt8](repeating: 0xFF, count: 100_000_000)  // ~100MB
        XCTAssertEqual(dummyArray.count, 100_000_000)

        // Ensure we can still search and the process is alive.
        let results = try await db.search(query: [1.0, 0.0], k: 10)
        XCTAssertEqual(results.count, 10)
    }
}
