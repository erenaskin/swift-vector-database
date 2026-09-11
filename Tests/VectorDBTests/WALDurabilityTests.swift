import XCTest
@testable import VectorDB

final class WALDurabilityTests: XCTestCase {
    
    var dbURL: URL!
    
    override func setUp() {
        super.setUp()
        let tempDir = FileManager.default.temporaryDirectory
        dbURL = tempDir.appendingPathComponent(UUID().uuidString + ".vdb")
    }
    
    override func tearDown() {
        try? FileManager.default.removeItem(at: dbURL)
        try? FileManager.default.removeItem(at: dbURL.appendingPathExtension("wal"))
        try? FileManager.default.removeItem(at: dbURL.appendingPathExtension("tmp"))
        super.tearDown()
    }
    
    func testFsyncDefaultIntervalRecoversWithoutSave() async throws {
        // walFlushInterval defaults to 1
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        try await db.save() // Create base file so recovery works
        
        try await db.insert(id: "v1", vector: [1.0, 1.0])
        
        // Recover without calling save(), tests that WAL is properly fsync'd and readable by a new instance
        let recoveredDB = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        
        let mirror = Mirror(reflecting: recoveredDB)
        guard let engine = mirror.children.first(where: { $0.label == "engine" })?.value as? Engine else {
            XCTFail("Could not access engine")
            return
        }
        
        let liveCount = engine.exportHNSWIndex().count
        XCTAssertEqual(liveCount, 1, "The record should be fully recoverable since fsync was called immediately.")
    }
    
    func testFsyncCallCountMatchesInterval() async throws {
        // Interval = 3
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL, walFlushInterval: 3)
        try await db.save() // Initialize files
        
        let mirror = Mirror(reflecting: db)
        guard let pm = mirror.children.first(where: { $0.label == "persistenceManager" })?.value as? PersistenceManager,
              let wal = pm.wal else {
            XCTFail("Could not access WAL")
            return
        }
        
        let initialFsyncCount = wal.fsyncCallCount
        
        // 7 inserts
        for i in 1...7 {
            try await db.insert(id: "v\(i)", vector: [Float(i), Float(i)])
        }
        
        // Expect 2 fsyncs (at 3 and 6)
        XCTAssertEqual(wal.fsyncCallCount - initialFsyncCount, 2, "fsync should be called exactly 2 times for 7 inserts with interval 3")
    }
    
    func testBatchInsertPerformsOnlyOneFsync() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL, walFlushInterval: 1)
        try await db.save()
        
        let mirror = Mirror(reflecting: db)
        guard let pm = mirror.children.first(where: { $0.label == "persistenceManager" })?.value as? PersistenceManager,
              let wal = pm.wal else {
            XCTFail("Could not access WAL")
            return
        }
        
        let initialFsyncCount = wal.fsyncCallCount
        
        var batch: [(id: String, vector: [Float], metadata: [String: String]?)] = []
        for i in 1...50 {
            batch.append(("v\(i)", [1.0, 1.0], nil))
        }
        
        try await db.batchInsert(batch)
        
        XCTAssertEqual(wal.fsyncCallCount - initialFsyncCount, 1, "batchInsert must perform exactly 1 fsync after completing, regardless of item count")
        
        let stats = await db.stats()
        XCTAssertEqual(stats.liveCount, 50)
    }
}
