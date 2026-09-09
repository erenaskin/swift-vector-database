import XCTest
@testable import VectorDB

final class FuzzTests: XCTestCase {
    
    /// Property-based fuzzing test simulating arbitrary sequences of inserts, updates, deletes, and searches.
    /// A `FlatIndex` acts as the ground-truth oracle.
    /// An `HNSWIndex` acts as the system under test.
    /// The test asserts that after any sequence of operations, HNSW provides correct results within the recall tolerance.
    func testStateConsistencyUnderFuzzing() throws {
        let dimension = 4
        let metric = DistanceMetric.dotProduct
        
        var flat = FlatIndex(dimension: dimension, metric: metric)
        var hnsw = HNSWIndex(dimension: dimension, metric: metric, params: HNSWParameters(M: 16))
        
        var rng = SeedableRNG(seed: 0xF002_BA11)
        
        var activeIDs = Set<Int32>()
        var maxInternalID: Int32 = 0
        
        let numOperations = 500
        
        for i in 0..<numOperations {
            let op = rng.nextFloat()
            
            if op < 0.6 || activeIDs.isEmpty {
                // 60% chance: Insert
                maxInternalID += 1
                let id = maxInternalID
                let vec = (0..<dimension).map { _ in rng.nextFloat() * 2.0 - 1.0 }
                
                try vec.withUnsafeBufferPointer { buf in
                    try flat.insert(internalID: id, vector: buf.baseAddress!)
                    try hnsw.insert(internalID: id, vector: buf.baseAddress!)
                }
                activeIDs.insert(id)
                
            } else if op < 0.8 {
                // 20% chance: Delete
                let idToDelete = activeIDs.randomElement()!
                try flat.remove(internalID: idToDelete)
                try hnsw.remove(internalID: idToDelete)
                activeIDs.remove(idToDelete)
                
            } else {
                // 20% chance: Update (simulated as Delete + Insert of new ID)
                let idToDelete = activeIDs.randomElement()!
                try flat.remove(internalID: idToDelete)
                try hnsw.remove(internalID: idToDelete)
                activeIDs.remove(idToDelete)
                
                maxInternalID += 1
                let id = maxInternalID
                let vec = (0..<dimension).map { _ in rng.nextFloat() * 2.0 - 1.0 }
                
                try vec.withUnsafeBufferPointer { buf in
                    try flat.insert(internalID: id, vector: buf.baseAddress!)
                    try hnsw.insert(internalID: id, vector: buf.baseAddress!)
                }
                activeIDs.insert(id)
            }
            
            // Periodically verify state
            if i % 100 == 0 || i == numOperations - 1 {
                XCTAssertEqual(flat.count, activeIDs.count)
                XCTAssertEqual(hnsw.count - hnsw.tombstoned.count, activeIDs.count)
                
                if !activeIDs.isEmpty {
                    // Search
                    let query = (0..<dimension).map { _ in rng.nextFloat() * 2.0 - 1.0 }
                    let k = min(10, activeIDs.count)
                    
                    let flatResults = query.withUnsafeBufferPointer { buf in
                        flat.search(query: buf.baseAddress!, k: k, ef: nil)
                    }
                    
                    let hnswResults = query.withUnsafeBufferPointer { buf in
                        hnsw.search(query: buf.baseAddress!, k: k, ef: 50)
                    }
                    
                    let flatIDs = Set(flatResults.map { $0.id })
                    let hnswIDs = Set(hnswResults.map { $0.id })
                    
                    let intersection = flatIDs.intersection(hnswIDs)
                    let recall = Double(intersection.count) / Double(k)
                    
                    // For such small datasets and small M, recall should be near perfect (1.0)
                    XCTAssertGreaterThanOrEqual(recall, 0.9, "Recall fell below threshold during fuzzing at op \(i)")
                }
            }
        }
    }
}
