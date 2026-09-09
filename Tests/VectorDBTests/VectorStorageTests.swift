/// VectorStorageTests.swift — Phase 3 Definition of Done tests (§7).
///
/// Coverage:
///   [DoD-1] Stress test: interleaving appends and reads to verify `grow()`
///           preserves all existing data byte-for-byte.
///   [DoD-2] Regression test: specific "dangling pointer across grow()" check.
///           We capture a pointer, force a resize, then verify that calling
///           `pointer(toSlot:)` again fetches the correct data from the *new*
///           allocation, proving we aren't caching pointers across `grow()`.
///   [DoD-3] Swap-remove correctness (tested via FlatIndexTests which now
///           uses VectorStorage under the hood).
///
/// NOTE: The 1M vector leak test is implemented in the benchmark runner
/// (`main.swift`) so it can be profiled under Instruments in Release mode.

import XCTest
@testable import VectorDB

final class VectorStorageTests: XCTestCase {

    // MARK: [DoD-1] Grow / Interleave Stress Test

    func testGrowPreservesDataByteForByte() {
        let dim = 16
        // Start with a tiny capacity so we force many `grow()` calls.
        let storage = VectorStorage(dimension: dim, initialCapacity: 2)
        
        let targetCount = 100
        var reference: [[Float]] = []
        var rng = SeedableRNG(seed: 0x920B_0001)
        
        for i in 0..<targetCount {
            // 1. Generate random vector
            let vector = (0..<dim).map { _ in rng.nextFloat() }
            reference.append(vector)
            
            // 2. Append to storage (will trigger multiple grows over the loop)
            _ = vector.withUnsafeBufferPointer { buf in
                storage.append(buf.baseAddress!)
            }
            
            // 3. Immediately read back ALL previously inserted vectors to prove
            //    that `grow()` didn't corrupt or drop any old data.
            for j in 0...i {
                let storedPtr = storage.pointer(toSlot: j)
                let refVec = reference[j]
                
                for d in 0..<dim {
                    XCTAssertEqual(storedPtr[d], refVec[d],
                        "Data corruption at slot \(j), dimension \(d) after \(i) total inserts")
                }
            }
        }
        
        XCTAssertEqual(storage.count, targetCount)
        // initialCapacity=2, doubled to 4, 8, 16, 32, 64, 128.
        XCTAssertEqual(storage.capacity, 128)
    }

    // MARK: [DoD-2] Dangling Pointer Regression Test

    func testDanglingPointerAcrossGrowRegression() {
        let dim = 4
        let storage = VectorStorage(dimension: dim, initialCapacity: 2)
        
        let v0: [Float] = [1, 2, 3, 4]
        let v1: [Float] = [5, 6, 7, 8]
        
        _ = v0.withUnsafeBufferPointer { storage.append($0.baseAddress!) }
        _ = v1.withUnsafeBufferPointer { storage.append($0.baseAddress!) }
        
        // storage is now full (count=2, capacity=2).
        
        // DANGER: We capture the raw pointer to slot 0.
        // In real code, we must NEVER do this across an append.
        let danglingPtr = storage.pointer(toSlot: 0)
        
        // Verify it's correct right now.
        XCTAssertEqual(danglingPtr[0], 1)
        
        // Force a resize (capacity 2 -> 4).
        let v2: [Float] = [9, 10, 11, 12]
        _ = v2.withUnsafeBufferPointer { storage.append($0.baseAddress!) }
        
        // Now, we fetch a FRESH pointer to slot 0.
        let freshPtr = storage.pointer(toSlot: 0)
        
        // The fresh pointer must have the correct data.
        XCTAssertEqual(freshPtr[0], 1.0)
        XCTAssertEqual(freshPtr[1], 2.0)
        XCTAssertEqual(freshPtr[2], 3.0)
        XCTAssertEqual(freshPtr[3], 4.0)
        
        // (Optional/Informational): The danglingPtr now points to freed memory.
        // We cannot XCTAssert on danglingPtr's contents without invoking undefined
        // behavior that might randomly pass or crash the test runner. The point
        // of this test is proving `freshPtr` is correct, and that `VectorStorage`
        // successfully migrated the data.
        XCTAssertNotEqual(freshPtr, danglingPtr, "VectorStorage must have moved to a new allocation after grow()")
    }
}
