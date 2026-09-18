import Foundation
import XCTest

@testable import VectorDatabase

/// NOTE (fix S1): these tests used to call `PersistenceManager.save(snapshot:idMap:)`,
/// which production code stopped using once `beginSave`/`finishSave` landed — so the
/// suite was validating a code path the app never executed, including a second,
/// independently written implementation of the file checksum. That path is gone;
/// every test here now goes through `saveSynchronously`, which is literally the same
/// three calls `VectorDatabase.save()` makes.
final class PersistenceTests: XCTestCase {
    var tempDir: URL!
    var dbURL: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try! FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        dbURL = tempDir.appendingPathComponent("test.vdb")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        super.tearDown()
    }

    func testCrashMidSaveLeavesOldSnapshotIntact() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        var index1 = HNSWIndex(dimension: 2, metric: .dotProduct)
        var map1 = IDMap()

        let v1: [Float] = [1.0, 0.0]
        let id1 = try map1.assign(externalID: "vec1")
        try index1.insert(internalID: id1, vector: v1)

        try manager.saveSynchronously(engine: Engine(hnswIndex: index1), idMap: map1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dbURL.path))

        // Simulate a crash that left a half-written .tmp behind.
        let tmpURL = dbURL.appendingPathExtension("tmp")
        try Data("garbage incomplete data".utf8).write(to: tmpURL)

        let manager2 = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        guard
            let (loadedIndex, loadedMap) = try manager2.load(
                fallbackMetric: .dotProduct, fallbackParams: .default)
        else {
            XCTFail("Failed to load")
            return
        }

        XCTAssertEqual(loadedIndex.count, 1, "The old snapshot should remain perfectly intact")
        XCTAssertEqual(loadedMap.internalID(for: "vec1"), 0)
    }

    func testCorruptedFileThrowsTypedErrorInsteadOfCrashing() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        var index = HNSWIndex(dimension: 2, metric: .dotProduct)
        var map = IDMap()

        // Give the file a non-empty body so there is something to corrupt.
        let v1: [Float] = [1.0, 0.0]
        let id1 = try map.assign(externalID: "vec1")
        try index.insert(internalID: id1, vector: v1)

        try manager.saveSynchronously(engine: Engine(hnswIndex: index), idMap: map)

        // Read the saved file
        var fileData = try Data(contentsOf: dbURL)

        // Corrupt bytes inside the body (header is 128 bytes).
        // Use bitwise NOT (~) instead of a fixed value like 0xFF: VectorStorage
        // allocates uninitialized memory, so reused memory blocks may already
        // contain 0xFF — making a fixed-value "corruption" a no-op. Bitwise NOT
        // guarantees every byte changes regardless of its original value.
        let corruptEnd = min(250, fileData.count)
        XCTAssertGreaterThan(corruptEnd, 150, "File is unexpectedly small to corrupt")
        for i in 150..<corruptEnd {
            fileData[i] = ~fileData[i]
        }

        // Write to a new file to avoid any inode/mmap cache issues
        let corruptedDBURL = tempDir.appendingPathComponent(UUID().uuidString + ".vdb")
        try fileData.write(to: corruptedDBURL, options: .atomic)

        let corruptManager = try PersistenceManager(databaseURL: corruptedDBURL, dimension: 2)

        do {
            _ = try corruptManager.load(fallbackMetric: .dotProduct, fallbackParams: .default)
            XCTFail("Should have thrown an error")
        } catch VectorDatabaseError.corruptFile(let reason) {
            XCTAssertTrue(reason.contains("Checksum validation failed"))
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }

    func testNewerFormatVersionThrows() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        let index = HNSWIndex(dimension: 2, metric: .dotProduct)
        let map = IDMap()
        try manager.saveSynchronously(engine: Engine(hnswIndex: index), idMap: map)

        let fh = try FileHandle(forUpdating: dbURL)
        var newFormat: UInt32 = 999
        let newFormatData = Data(bytes: &newFormat, count: 4)
        // Format version is at offset 4 in the header (magic is 0-3)
        try fh.seek(toOffset: 4)
        try fh.write(contentsOf: newFormatData)
        try fh.close()

        // The version check runs before the checksum check, so we expect the
        // version error rather than a corruption error.
        do {
            _ = try manager.load(fallbackMetric: .dotProduct, fallbackParams: .default)
            XCTFail("Should have thrown an error")
        } catch VectorDatabaseError.unsupportedFileVersion(let found, let supported) {
            XCTAssertEqual(found, 999)
            XCTAssertEqual(supported, 2)
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }

    /// Fix K4 coverage: opening an existing store with the wrong dimension must
    /// fail loudly instead of silently mis-parsing WAL records.
    func testLoadWithMismatchedDimensionThrows() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        var index = HNSWIndex(dimension: 2, metric: .dotProduct)
        var map = IDMap()
        let id = try map.assign(externalID: "vec1")
        try index.insert(internalID: id, vector: [1.0, 0.0])
        try manager.saveSynchronously(engine: Engine(hnswIndex: index), idMap: map)

        let wrongManager = try PersistenceManager(databaseURL: dbURL, dimension: 8)
        do {
            _ = try wrongManager.load(fallbackMetric: .dotProduct, fallbackParams: .default)
            XCTFail("Should have thrown dimensionMismatch")
        } catch VectorDatabaseError.dimensionMismatch(let expected, let got) {
            XCTAssertEqual(expected, 2)
            XCTAssertEqual(got, 8)
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }

    func testWALRecovery() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        let index = HNSWIndex(dimension: 2, metric: .dotProduct)
        let map = IDMap()
        try manager.saveSynchronously(engine: Engine(hnswIndex: index), idMap: map)

        let vec: [Float] = [0.5, 0.5]
        let record = WALRecord(
            opcode: .insert, internalID: 99, externalID: "ext99", timestamp: 12345, vectorData: vec)
        try manager.wal?.append(record: record)
        try manager.wal?.fsync()

        guard
            let (loadedIndex, _) = try manager.load(
                fallbackMetric: .dotProduct, fallbackParams: .default)
        else {
            XCTFail("Failed to load")
            return
        }

        XCTAssertEqual(loadedIndex.count, 1)

        // Verify we can search for it (meaning it reached GraphStorage successfully)
        let results = loadedIndex.search(query: vec, k: 1)
        XCTAssertEqual(results.first?.id, 99)
    }

    func testAppKilledMidWALWriteSkipsIncompleteRecord() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        let index = HNSWIndex(dimension: 2, metric: .dotProduct)
        let map = IDMap()
        try manager.saveSynchronously(engine: Engine(hnswIndex: index), idMap: map)

        // Write a complete record
        let vec: [Float] = [0.5, 0.5]
        let record = WALRecord(
            opcode: .insert, internalID: 99, externalID: "ext99", timestamp: 12345, vectorData: vec)
        try manager.wal?.append(record: record)
        try manager.wal?.fsync()

        // Manually write an incomplete garbage record to the WAL
        let walURL = dbURL.appendingPathExtension("wal")
        let fh = try FileHandle(forUpdating: walURL)
        try fh.seekToEnd()
        // Write exactly 3 bytes (incomplete record)
        try fh.write(contentsOf: Data([0x00, 0x01, 0x02]))
        try fh.close()

        // Recovery should gracefully skip the trailing incomplete record and load the valid one
        let manager2 = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        guard
            let (loadedIndex, _) = try manager2.load(
                fallbackMetric: .dotProduct, fallbackParams: .default)
        else {
            XCTFail("Failed to load")
            return
        }
        XCTAssertEqual(loadedIndex.count, 1)  // 99 was loaded successfully
    }

    /// Fix K1 coverage: a record appended WHILE a save is in flight must survive
    /// the WAL truncation that the save performs when it completes.
    func testWALRecordsWrittenDuringSaveSurviveTruncation() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        var index = HNSWIndex(dimension: 2, metric: .dotProduct)
        var map = IDMap()
        let id = try map.assign(externalID: "snapshotted")
        try index.insert(internalID: id, vector: [1.0, 0.0])

        // 1. Take the snapshot and its WAL watermark.
        let pending = try manager.beginSave(engine: Engine(hnswIndex: index))

        // 2. Simulate the actor appending a brand-new record while the detached
        //    save task is still running.
        try manager.wal?.append(
            record: WALRecord(
                opcode: .insert, internalID: 77, externalID: "duringSave",
                timestamp: 1, vectorData: [0.0, 1.0]))
        try manager.wal?.fsync()

        // 3. Finish the save and reclaim only what the snapshot superseded.
        try manager.finishSave(pending, idMap: map)
        try manager.truncateWALAfterSave(upTo: pending.walTruncationOffset)

        // 4. The in-flight record must still be replayable.
        let survivors = try XCTUnwrap(manager.wal?.readAll())
        XCTAssertEqual(
            survivors.count, 1, "The record appended during save was erased by truncation")
        XCTAssertEqual(survivors.first?.externalID, "duringSave")

        let reopened = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        guard
            let (loadedIndex, loadedMap) = try reopened.load(
                fallbackMetric: .dotProduct, fallbackParams: .default)
        else {
            XCTFail("Failed to load")
            return
        }
        XCTAssertEqual(loadedIndex.count, 2)
        XCTAssertNotNil(loadedMap.internalID(for: "snapshotted"))
        XCTAssertNotNil(loadedMap.internalID(for: "duringSave"))
    }

    // MARK: - Mapped growth

    func testMappedVectorDatabaseGrowDoesNotCrash() async throws {
        // 1. First instance: create and save with some capacity.
        // Default capacity is 1024. We insert 1025 to force a growth to 2048 in heap mode,
        // then save it.
        let db1 = try VectorDatabase(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 1...1025 {
            try await db1.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        try await db1.save()
        await db1.close()

        // 2. Second instance: open mapped, then insert enough to trigger grow() while mapped.
        let db2 = try VectorDatabase(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 1026...2049 {
            try await db2.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }

        let results2 = try await db2.search(query: [2049.0, 2049.0], k: 1)
        XCTAssertEqual(
            results2.first?.id, "v2049", "Should find newly inserted vector after mapped growth")

        try await db2.save()
        await db2.close()

        // 3. Third instance: open again and verify.
        let db3 = try VectorDatabase(dimension: 2, metric: .euclidean, path: dbURL)
        let results3 = try await db3.search(query: [2049.0, 2049.0], k: 1)
        XCTAssertEqual(
            results3.first?.id, "v2049", "Should survive a full round-trip after mapped growth")

        let results1 = try await db3.search(query: [1.0, 1.0], k: 1)
        XCTAssertEqual(results1.first?.id, "v1", "Original vectors should still be searchable")
        await db3.close()
    }

    func testMappedGraphStorageGrowWithUpperLayers() async throws {
        // HNSW threshold is 2000. We need > 2000 vectors to build upper layers.
        let db1 = try VectorDatabase(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 1...2049 {
            try await db1.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        try await db1.save()
        await db1.close()

        // Open mapped, insert past the persisted capacity to trigger GraphStorage
        // grow() on mapped upper layers.
        let db2 = try VectorDatabase(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 2050...4097 {
            try await db2.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }

        let results2 = try await db2.search(query: [4097.0, 4097.0], k: 1)
        XCTAssertEqual(results2.first?.id, "v4097")

        try await db2.save()
        await db2.close()

        let db3 = try VectorDatabase(dimension: 2, metric: .euclidean, path: dbURL)
        let results3 = try await db3.search(query: [4097.0, 4097.0], k: 1)
        XCTAssertEqual(results3.first?.id, "v4097")
        await db3.close()
    }
}
