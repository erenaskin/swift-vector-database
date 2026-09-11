import XCTest
@testable import VectorDB

final class InspectorTests: XCTestCase {
    
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
        let walURL = dbURL.appendingPathExtension("wal")
        if FileManager.default.fileExists(atPath: walURL.path) {
            try? FileManager.default.removeItem(at: walURL)
        }
        super.tearDown()
    }
    
    // 1. Flat mode behavior: Returns nil / [] safely
    func testFlatModeInspector() async throws {
        // HNSW Threshold is 2000 by default. So 2 items will use FlatIndex.
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        try await db.insert(id: "doc1", vector: [1.0, 1.0])
        try await db.insert(id: "doc2", vector: [2.0, 2.0])
        
        let inspector = db.inspector
        
        let ep = await inspector.entryPointID()
        XCTAssertNil(ep, "Flat modda entry point nil olmali")
        
        let level = await inspector.nodeLevel(of: "doc1")
        XCTAssertNil(level, "Flat modda node level nil olmali")
        
        let layers = await inspector.layerCount(of: "doc1")
        XCTAssertNil(layers, "Flat modda layer count nil olmali")
        
        let neighbors = await inspector.neighbors(of: "doc1", atLayer: 0)
        XCTAssertNil(neighbors, "Flat modda neighbors nil olmali")
    }
    
    // 2. HNSW Mode: Normal read operations
    func testHNSWModeInspector() async throws {
        // Set threshold to 2 so we immediately migrate to HNSW on 3rd insert.
        let db = try VectorDB(dimension: 2, metric: .dotProduct, parameters: .default, path: dbURL)
        
        // Let's insert a bunch to ensure HNSW is built
        // To force HNSW quickly, we can manually initialize the Engine? 
        // Actually, threshold is hardcoded in VectorDB.init to 2000. 
        // We can't change the threshold easily from public VectorDB init because hnswThreshold is not exposed.
        // Let's just create an IndexRouter directly, or insert 2001 items.
        // Wait, inserting 2001 items takes a second, totally fine.
        for i in 0...2005 {
            try await db.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }
        
        let inspector = db.inspector
        
        let ep = await inspector.entryPointID()
        XCTAssertNotNil(ep, "HNSW modda entry point nil OLMAMALI")
        
        guard let epID = ep else { return }
        
        let epLevel = await inspector.nodeLevel(of: epID)
        XCTAssertNotNil(epLevel)
        
        let epLayerCount = await inspector.layerCount(of: epID)
        XCTAssertEqual(epLayerCount, epLevel! + 1)
        
        // neighbors check
        let neighborsL0 = await inspector.neighbors(of: epID, atLayer: 0)
        XCTAssertNotNil(neighborsL0)
        
        // Invalid layer query (above level) -> nil
        let invalidLayerNeighbors = await inspector.neighbors(of: epID, atLayer: epLevel! + 1)
        XCTAssertNil(invalidLayerNeighbors)
    }
    
    // 3. Tombstone Filtering and Behavior
    func testTombstoneFiltering() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        for i in 0...2005 {
            try await db.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }
        
        let inspector = db.inspector
        guard let epID = await inspector.entryPointID() else {
            XCTFail("Entry point bulunamadi")
            return
        }
        
        // Find a neighbor of the entry point at layer 0
        guard let neighbors = await inspector.neighbors(of: epID, atLayer: 0), !neighbors.isEmpty else {
            XCTFail("Komsu bulunamadi")
            return
        }
        
        let firstNeighbor = neighbors.first!
        
        // Before delete, it is a neighbor
        let neighborsBefore = await inspector.neighbors(of: epID, atLayer: 0)
        XCTAssertTrue(neighborsBefore!.contains(firstNeighbor))
        
        // Now delete this neighbor
        try await db.delete(id: firstNeighbor)
        
        // 1. nodeLevel for deleted node should be nil
        let deletedLevel = await inspector.nodeLevel(of: firstNeighbor)
        XCTAssertNil(deletedLevel)
        
        // 2. neighbors(of:) for deleted node should be nil
        let deletedNeighbors = await inspector.neighbors(of: firstNeighbor, atLayer: 0)
        XCTAssertNil(deletedNeighbors)
        
        // 3. MOST IMPORTANT: It MUST be filtered out of the entryPoint's neighbors!
        let neighborsAfter = await inspector.neighbors(of: epID, atLayer: 0)
        XCTAssertNotNil(neighborsAfter)
        XCTAssertFalse(neighborsAfter!.contains(firstNeighbor), "Tombstone filtrelemesi calismiyor! Silinmis node hala komsu listesinde.")
    }
    
    // 4. Entry Point Deletion Fallback (Inspector Mirroring)
    func testEntryPointDeletion() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        for i in 0...2005 {
            try await db.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }
        
        let inspector = db.inspector
        guard let epID1 = await inspector.entryPointID() else {
            XCTFail("Entry point bulunamadi")
            return
        }
        
        // Delete the entry point itself
        try await db.delete(id: epID1)
        
        // Inspector should immediately reflect the new entry point from HNSWIndex.
        let epID2 = await inspector.entryPointID()
        XCTAssertNotNil(epID2)
        XCTAssertNotEqual(epID1, epID2, "Entry point silinince HNSW yeni bir tane secmeli ve inspector bunu yansitmali")
        
        let level1 = await inspector.nodeLevel(of: epID1)
        XCTAssertNil(level1, "Silinmis entry point artik yok")
    }
}
