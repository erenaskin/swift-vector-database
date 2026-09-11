import XCTest
@testable import VectorDB
import Foundation

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
        
        try manager.save(index: index1, idMap: map1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dbURL.path))
        
        let tmpURL = dbURL.appendingPathExtension("tmp")
        try Data("garbage incomplete data".utf8).write(to: tmpURL)
        
        let manager2 = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        guard let (loadedIndex, loadedMap) = try manager2.load() else {
            XCTFail("Failed to load")
            return
        }
        
        XCTAssertEqual(loadedIndex.count, 1, "The old snapshot should remain perfectly intact")
        XCTAssertEqual(loadedMap.internalID(for: "vec1"), 0)
    }
    
    func testCorruptedFileThrowsTypedErrorInsteadOfCrashing() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        let index = HNSWIndex(dimension: 2, metric: .dotProduct)
        let map = IDMap()
        try manager.save(index: index, idMap: map)
        
        // Read the saved file
        var fileData = try Data(contentsOf: dbURL)
        
        // Corrupt bytes inside the body (header is 128 bytes).
        // Use bitwise NOT (~) instead of a fixed value like 0xFF.
        // VectorStorage allocates uninitialized memory, so after 91 tests of
        // heavy allocation/deallocation, reused memory blocks may already contain
        // 0xFF — making a fixed-value "corruption" a no-op. Bitwise NOT
        // guarantees every byte changes regardless of its original value.
        for i in 150..<250 {
            fileData[i] = ~fileData[i]
        }
        
        // Write to a new file to avoid any inode/mmap cache issues
        let corruptedDBURL = tempDir.appendingPathComponent(UUID().uuidString + ".vdb")
        try fileData.write(to: corruptedDBURL, options: .atomic)
        
        let corruptManager = try PersistenceManager(databaseURL: corruptedDBURL, dimension: 2)
        
        do {
            let _ = try corruptManager.load()
            XCTFail("Should have thrown an error")
        } catch VectorDBError.corruptFile(let reason) {
            XCTAssertTrue(reason.contains("Checksum validation failed"))
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }
    

    func testNewerFormatVersionThrows() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        let index = HNSWIndex(dimension: 2, metric: .dotProduct)
        let map = IDMap()
        try manager.save(index: index, idMap: map)
        
        let fh = try FileHandle(forUpdating: dbURL)
        var newFormat: UInt32 = 999
        let newFormatData = Data(bytes: &newFormat, count: 4)
        // Format version is at offset 4 in header (magic is 0-3)
        try fh.seek(toOffset: 4)
        try fh.write(contentsOf: newFormatData)
        
        // We also need to fix the checksum so it doesn't fail checksum first
        // But the code checks format version before checksum, so it should throw format version error first!
        try fh.close()
        
        do {
            let _ = try manager.load()
            XCTFail("Should have thrown an error")
        } catch VectorDBError.unsupportedFileVersion(let found, let supported) {
            XCTAssertEqual(found, 999)
            XCTAssertEqual(supported, 1)
        } catch {
            XCTFail("Threw unexpected error: \(error)")
        }
    }
    
    func testWALRecovery() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        let index = HNSWIndex(dimension: 2, metric: .dotProduct)
        let map = IDMap()
        try manager.save(index: index, idMap: map)
        
        let vec: [Float] = [0.5, 0.5]
        let record = WALRecord(opcode: .insert, internalID: 99, timestamp: 12345, vectorData: vec)
        try manager.wal?.append(record: record)
        try manager.wal?.fsync()
        
        guard let (loadedIndex, _) = try manager.load() else {
            XCTFail("Failed to load")
            return
        }
        
        XCTAssertEqual(loadedIndex.count, 1)
        
        // Let's verify we can search for it (meaning it was inserted to GraphStorage successfully)
        // Let's verify we can search for it (meaning it was inserted to GraphStorage successfully)
        let results = loadedIndex.search(query: vec, k: 1)
        XCTAssertEqual(results.first?.id, 99)
    }
    
    func testAppKilledMidWALWriteSkipsIncompleteRecord() throws {
        let manager = try PersistenceManager(databaseURL: dbURL, dimension: 2)
        let index = HNSWIndex(dimension: 2, metric: .dotProduct)
        let map = IDMap()
        try manager.save(index: index, idMap: map)
        
        // Write a complete record
        let vec: [Float] = [0.5, 0.5]
        let record = WALRecord(opcode: .insert, internalID: 99, timestamp: 12345, vectorData: vec)
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
        guard let (loadedIndex, _) = try manager2.load() else {
            XCTFail("Failed to load")
            return
        }
        XCTAssertEqual(loadedIndex.count, 1) // 99 was loaded successfully
    }
    
    // MARK: - Task 5 Tests
    
    func testMappedVectorDBGrowDoesNotCrash() async throws {
        // 1. First instance: create and save with some capacity
        // Default capacity is 1024. We insert 1025 to force a growth to 2048 in heap mode,
        // then save it. The saved file will have capacity 2048.
        let db1 = try VectorDB(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 1...1025 {
            try await db1.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        try await db1.save()
        await db1.close()
        
        // 2. Second instance: open mapped, then insert enough to exceed 2048
        // We will insert up to 2049. This triggers grow() while mapped.
        let db2 = try VectorDB(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 1026...2049 {
            try await db2.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        
        // Assert we can search a new vector
        let results2 = try await db2.search(query: [2049.0, 2049.0], k: 1)
        XCTAssertEqual(results2.first?.id, "v2049", "Should find newly inserted vector after mapped growth")
        
        // Save and close
        try await db2.save()
        await db2.close()
        
        // 3. Third instance: open again and verify
        let db3 = try VectorDB(dimension: 2, metric: .euclidean, path: dbURL)
        let results3 = try await db3.search(query: [2049.0, 2049.0], k: 1)
        XCTAssertEqual(results3.first?.id, "v2049", "Should survive a full round-trip after mapped growth")
        
        let results1 = try await db3.search(query: [1.0, 1.0], k: 1)
        XCTAssertEqual(results1.first?.id, "v1", "Original vectors should still be searchable")
        await db3.close()
    }
    
    func testMappedGraphStorageGrowWithUpperLayers() async throws {
        // HNSW threshold is 2000. We need to insert > 2000 vectors to build upper layers.
        // We insert 2049 vectors. Capacity becomes 4096.
        let db1 = try VectorDB(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 1...2049 {
            try await db1.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        try await db1.save()
        await db1.close()
        
        // Open mapped, insert to exceed 4096 capacity.
        // We will insert up to 4097. This triggers GraphStorage grow() on mapped upper layers!
        let db2 = try VectorDB(dimension: 2, metric: .euclidean, path: dbURL)
        for i in 2050...4097 {
            try await db2.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        
        // Verify search uses the graph properly (HNSW search)
        let results2 = try await db2.search(query: [4097.0, 4097.0], k: 1)
        XCTAssertEqual(results2.first?.id, "v4097")
        
        try await db2.save()
        await db2.close()
        
        // Open third instance
        let db3 = try VectorDB(dimension: 2, metric: .euclidean, path: dbURL)
        let results3 = try await db3.search(query: [4097.0, 4097.0], k: 1)
        XCTAssertEqual(results3.first?.id, "v4097")
        await db3.close()
    }
}
