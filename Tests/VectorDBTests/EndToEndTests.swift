import XCTest
@testable import VectorDB

final class EndToEndTests: XCTestCase {
    var dbURL: URL!
    
    override func setUp() {
        super.setUp()
        dbURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("vdb")
    }
    
    override func tearDown() {
        try? FileManager.default.removeItem(at: dbURL)
        try? FileManager.default.removeItem(at: dbURL.appendingPathExtension("wal"))
        super.tearDown()
    }
    
    // A) UÇTAN UCA KARMA SENARYO TESTİ
    func testEndToEndMixedScenario() async throws {
        // 1. Insert 30 vectors to ensure we pass HNSW threshold (if we set it low, 
        // but default is 2000. For test, let's insert 2005)
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        
        for i in 0...2005 {
            try await db.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }
        
        // 2. Update metadata
        try await db.updateMetadata(id: "doc10", metadata: ["updated": "true"])
        try await db.updateMetadata(id: "doc20", metadata: ["status": "active"])
        
        // 3. Delete some nodes (including entry point if possible)
        let inspector = db.inspector
        guard let ep1 = await inspector.entryPointID() else {
            XCTFail("No entry point")
            return
        }
        
        try await db.delete(id: "doc5")
        try await db.delete(id: ep1) // Delete the entry point
        
        // 4. Query graph with inspector
        let ep2 = await inspector.entryPointID()
        XCTAssertNotNil(ep2)
        XCTAssertNotEqual(ep1, ep2, "Entry point should have changed")
        
        let deletedLevel = await inspector.nodeLevel(of: "doc5")
        XCTAssertNil(deletedLevel, "Deleted node should not have a level")
        
        let deletedNeighbors = await inspector.neighbors(of: "doc5", atLayer: 0)
        XCTAssertNil(deletedNeighbors, "Deleted node should not have neighbors")
        
        if let ep2 = ep2 {
            let ep2Neighbors = await inspector.neighbors(of: ep2, atLayer: 0) ?? []
            XCTAssertFalse(ep2Neighbors.contains("doc5"), "Deleted node should not appear in neighbors")
            XCTAssertFalse(ep2Neighbors.contains(ep1), "Deleted entry point should not appear in neighbors")
        }
        
        // 5. Verify listIDs and get
        let ids = await db.listIDs(offset: 0, limit: 3000)
        XCTAssertFalse(ids.contains("doc5"))
        XCTAssertFalse(ids.contains(ep1))
        
        let doc10 = await db.get(id: "doc10")
        XCTAssertEqual(doc10?.metadata?["updated"], "true")
        
        // 6. Close WITHOUT saving (to trigger WAL replay)
        await db.close()
        
        // 7. Reload and verify EVERYTHING again
        let dbReloaded = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        
        let idsReloaded = await dbReloaded.listIDs(offset: 0, limit: 3000)
        XCTAssertFalse(idsReloaded.contains("doc5"))
        XCTAssertFalse(idsReloaded.contains(ep1))
        XCTAssertTrue(idsReloaded.contains("doc10"))
        
        let doc10Reloaded = await dbReloaded.get(id: "doc10")
        // Check that metadata WAL insertion is NOT lost! (C4 Fix Validation)
        XCTAssertEqual(doc10Reloaded?.metadata?["updated"], "true")
        
        let inspectorReloaded = dbReloaded.inspector
        let epReloaded = await inspectorReloaded.entryPointID()
        XCTAssertEqual(epReloaded, ep2, "Entry point should be the same after replay")
        
        // 8. Insert a new node, likely to fall into a new layer, mixing mmap growth
        // Since we didn't save, it's actually in Heap mode! Wait.
        // If we close WITHOUT save, when it reopens it reads WAL. If the base file didn't exist, it stays in Heap mode!
        // Let's force it to be mapped first.
        try await dbReloaded.save()
        await dbReloaded.close()
        
        let dbMapped = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        for i in 2006...10000 {
            try await dbMapped.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }
        // Should not crash due to mmap fix.
    }
    
    // B) CONCURRENCY STRES TESTİ
    func testInspectorConcurrency() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        for i in 0...2005 {
            try await db.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }
        
        let inspector = db.inspector
        
        await withTaskGroup(of: Void.self) { group in
            for i in 2006...3000 {
                group.addTask {
                    try? await db.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
                }
                group.addTask {
                    try? await db.delete(id: "doc\(i-1000)")
                }
                group.addTask {
                    try? await db.updateMetadata(id: "doc\(i-500)", metadata: ["k": "v"])
                }
                group.addTask {
                    _ = try? await db.get(id: "doc\(i-10)")
                }
                group.addTask {
                    _ = await db.listIDs(offset: 0, limit: 10)
                }
                group.addTask {
                    _ = await inspector.entryPointID()
                    _ = await inspector.neighbors(of: "doc\(i-500)", atLayer: 0)
                }
            }
        }
        
        let ep = await inspector.entryPointID()
        XCTAssertNotNil(ep) // Should still be valid and not deadlocked
    }
}
