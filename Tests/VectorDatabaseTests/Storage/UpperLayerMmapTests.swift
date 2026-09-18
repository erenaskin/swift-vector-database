import XCTest

@testable import VectorDatabase

final class UpperLayerMmapTests: XCTestCase {
    var dbURL: URL!

    override func setUp() {
        super.setUp()
        dbURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("vdb")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: dbURL)
        try? FileManager.default.removeItem(at: dbURL.appendingPathExtension("wal"))
        super.tearDown()
    }

    // This test forces a scenario where an HNSW graph is memory-mapped,
    // and a subsequent insertion generates a level higher than the previously mapped maximum layer.
    // Without the fix, this will crash with a SIGBUS or silent memory corruption.
    func testMmapBoundaryExceededCrash() async throws {
        // Step 1: Create a DB and insert just enough to trigger HNSW (threshold 2000).
        let db1 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        for i in 0...2005 {
            try await db1.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }

        // At this point, the HNSW graph has a certain maximum layer (e.g. 2 or 3).
        // Save the index to disk.
        try await db1.save()
        await db1.close()

        // Step 2: Reload the DB. It will use .mapped allocation for GraphStorage.
        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        // Step 3: Insert more nodes. Eventually, one of these nodes will randomly
        // receive a level greater than the old maximum layer.
        // When this happens, GraphStorage.ensureUpperLayerAllocated will be called.
        // If the bug is present, it will append to neighborCounts but NOT allocate
        // new backing memory for the upperLayerPointer, causing a crash/SIGBUS.
        for i in 2006...12000 {
            try await db2.insert(id: "doc\(i)", vector: [Float(i), Float(i)])
        }

        // Step 4: Verify the graph is completely healthy and readable
        let inspector = db2.inspector
        guard let ep = await inspector.entryPointID() else {
            XCTFail("Should have an entry point")
            return
        }

        let epLevel = await inspector.nodeLevel(of: ep)
        XCTAssertNotNil(epLevel)

        // Ensure neighbors are readable at the highest layer
        let neighbors = await inspector.neighbors(of: ep, atLayer: epLevel!)
        XCTAssertNotNil(neighbors)

        // Pass if no crash occurred!
    }
}
