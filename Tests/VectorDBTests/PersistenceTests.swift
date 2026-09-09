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
}
