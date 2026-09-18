import XCTest
@testable import VectorDatabase

final class GraphStorageMmapTests: XCTestCase {
    var fileURL: URL!
    
    override func setUp() {
        super.setUp()
        fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("bin")
    }
    
    override func tearDown() {
        try? FileManager.default.removeItem(at: fileURL)
        super.tearDown()
    }
    
    func testMmapSIGBUSCrash() throws {
        // 1. Create a fake file with enough size for layer 0 and 1 upper layer.
        let m = 16
        let mMax0 = 32
        let capacity = 1024
        
        let layer0Size = capacity * mMax0 * MemoryLayout<Int32>.size
        let upperLayerSize = capacity * m * MemoryLayout<Int32>.size
        
        // Total file size allows for exactly Layer 0 and Layer 1.
        let totalSize = layer0Size + upperLayerSize
        
        // Create the file filled with zeros.
        let data = Data(count: totalSize)
        try data.write(to: fileURL)
        
        let mappedFile = try MappedFile(path: fileURL.path, initialSize: totalSize)
        
        // neighborCounts for layer 0 and layer 1
        let neighborCounts: [[Int]] = [Array(repeating: 0, count: capacity), Array(repeating: 0, count: capacity)]
        
        let storage = GraphStorage(
            m: m,
            mMax0: mMax0,
            capacity: capacity,
            count: 10,
            mappedFile: mappedFile,
            offset: 0,
            neighborCounts: neighborCounts
        )
        
        // 2. Request layer 2. Since mapped file only has size for layer 0 and 1,
        // without the fix, this will NOT allocate any new memory, but will just update neighborCounts.
        // It will return a pointer that points PAST the end of the mmap file.
        
        storage.setNeighbors(of: 0, at: 2, to: [1, 2, 3])
        
        // 3. Since the file is mapped with exactly totalSize bytes, writing to layer 2
        // offset should trigger a SIGBUS on Darwin.
        // If the fix is in place, it will transition to .heap, allocate new memory safely,
        // and NOT crash.

        // 4. Not crashing is necessary but not sufficient: the fallback must also
        // preserve the data it was asked to write. Read the neighbors back and
        // confirm they round-trip correctly, instead of only checking "did the
        // process survive" (the original version of this test had no assertion
        // at all past this point).
        XCTAssertEqual(
            storage.neighbors(of: 0, at: 2), [1, 2, 3],
            "Neighbors written to a layer past the mmap'd capacity must survive the "
            + "fallback-to-heap transition, not just avoid crashing.")
    }
}
