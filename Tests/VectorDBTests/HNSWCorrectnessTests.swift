/// HNSWCorrectnessTests.swift — Phase 4b unit tests (§8.5 DoD items).
///
/// Coverage:
///   [DoD-A] Heap ordering tested with KNOWN VALUES, independent of the rest of HNSW.
///           Verifies both the max-heap (candidates) and min-heap (found) directions
///           to catch the distance-minimization vs. similarity-maximization comparator
///           bug the guide warns about in §8.5.
///   [DoD-B] RNG determinism: same seed + same insert order = byte-identical
///           randomLevel() sequence and graph structure across two separate runs.

import XCTest
@testable import VectorDB

final class HNSWCorrectnessTests: XCTestCase {

    // MARK: [DoD-A] Heap ordering — independent of rest of HNSW

    /// Verifies the MAX-heap used for candidates:
    /// `BinaryHeap.maxByScore` must always pop the HIGHEST score first.
    /// In the paper this is "min-heap by distance" — we pop the CLOSEST node first.
    func testCandidateHeapPopsHighestScoreFirst() {
        var heap = BinaryHeap<Candidate>.maxByScore([
            Candidate(id: 10, score: 0.5),
            Candidate(id: 20, score: 0.9),
            Candidate(id: 30, score: 0.1),
            Candidate(id: 40, score: 0.7),
        ])

        // First pop must be the highest score.
        XCTAssertEqual(heap.pop()?.score, 0.9, "candidateHeap must pop HIGHEST score first (closest node)")
        XCTAssertEqual(heap.pop()?.score, 0.7)
        XCTAssertEqual(heap.pop()?.score, 0.5)
        XCTAssertEqual(heap.pop()?.score, 0.1, "candidateHeap must pop LOWEST score last (farthest node)")
        XCTAssertNil(heap.pop(), "empty heap must return nil")
    }

    /// Verifies the MIN-heap used for the result set:
    /// `BinaryHeap.minByScore` must always pop the LOWEST score first.
    /// In the paper this is "max-heap by distance" — we pop the FARTHEST node first,
    /// so it can be evicted when a better candidate arrives.
    func testFoundHeapPopsLowestScoreFirst() {
        var heap = BinaryHeap<Candidate>.minByScore([
            Candidate(id: 10, score: 0.5),
            Candidate(id: 20, score: 0.9),
            Candidate(id: 30, score: 0.1),
            Candidate(id: 40, score: 0.7),
        ])

        // First pop must be the lowest score (farthest node).
        XCTAssertEqual(heap.pop()?.score, 0.1, "foundHeap must pop LOWEST score first (farthest, to evict)")
        XCTAssertEqual(heap.pop()?.score, 0.5)
        XCTAssertEqual(heap.pop()?.score, 0.7)
        XCTAssertEqual(heap.pop()?.score, 0.9, "foundHeap must pop HIGHEST score last (closest kept)")
        XCTAssertNil(heap.pop(), "empty heap must return nil")
    }

    /// Verifies peek() does NOT remove the element.
    func testPeekDoesNotRemove() {
        let heap = BinaryHeap<Candidate>.maxByScore([
            Candidate(id: 1, score: 0.8),
            Candidate(id: 2, score: 0.3),
        ])
        XCTAssertEqual(heap.peek()?.score, 0.8)
        XCTAssertEqual(heap.peek()?.score, 0.8, "peek() must be non-destructive")
        XCTAssertEqual(heap.count, 2, "count must not change after peek()")
    }

    /// Verifies `sortedDescending()` produces highest-first order.
    func testSortedDescending() {
        let heap = BinaryHeap<Candidate>.maxByScore([
            Candidate(id: 1, score: 0.3),
            Candidate(id: 2, score: 0.9),
            Candidate(id: 3, score: 0.6),
        ])
        let sorted = heap.sortedDescending()
        XCTAssertEqual(sorted.map(\.score), [0.9, 0.6, 0.3], "sortedDescending must return highest score first")
    }

    /// Verifies build-heap from an array (O(N) heapify).
    func testBuildHeapFromArray() {
        var heap = BinaryHeap<Candidate>.maxByScore([
            Candidate(id: 5, score: 0.2),
            Candidate(id: 6, score: 0.8),
            Candidate(id: 7, score: 0.5),
        ])
        XCTAssertEqual(heap.pop()?.score, 0.8, "heapify from array must produce correct max-heap root")
    }

    /// Verifies tie-breaking doesn't crash (equal scores).
    func testEqualScoresDoNotCrash() {
        var heap = BinaryHeap<Candidate>.maxByScore([
            Candidate(id: 1, score: 0.5),
            Candidate(id: 2, score: 0.5),
            Candidate(id: 3, score: 0.5),
        ])
        XCTAssertEqual(heap.count, 3)
        _ = heap.pop()
        _ = heap.pop()
        _ = heap.pop()
        XCTAssertEqual(heap.count, 0, "all equal-score elements must be poppable without crash")
    }

    /// CRITICAL: Verifies that swapping the two heap orderings would produce
    /// the WRONG result — i.e., confirms we're not accidentally using the same
    /// heap in both roles. Uses the eviction logic of searchLayer:
    ///   - When found.count > ef, pop the FARTHEST (lowest score).
    ///   - If we accidentally used maxByScore for found, we'd evict the CLOSEST
    ///     (highest score) instead, silently destroying recall.
    func testEvictsFarthestNotClosest() {
        var foundHeap = BinaryHeap<Candidate>.minByScore()
        let ef = 2
        let incoming: [Float] = [0.9, 0.3, 0.7, 0.1, 0.5]
        for (i, s) in incoming.enumerated() {
            foundHeap.push(Candidate(id: Int32(i), score: s))
            if foundHeap.count > ef { foundHeap.pop() }
        }
        // With ef=2 and minByScore, the two lowest-score candidates are evicted each time
        // a new one exceeds the limit. Survivors must be the 2 highest scores: 0.9 and 0.7.
        // sortedDescending() pops from the min-heap lowest→highest, so result is ascending.
        // We sort again to get a canonical order for comparison.
        let survivors = foundHeap.sortedDescending().map(\.score).sorted(by: >)
        XCTAssertEqual(survivors, [0.9, 0.7],
            "foundHeap must evict FARTHEST (lowest score): bug if [0.9, 0.7] is not the result")
    }

    // MARK: [DoD-B] RNG determinism — same seed = identical level sequence

    func testRandomLevelDeterminism() {
        let params = HNSWParameters(M: 16, efConstruction: 100, efSearch: 64, seed: 0xDEAD_BEEF)
        var index1 = HNSWIndex(dimension: 4, metric: .dotProduct, params: params)
        var index2 = HNSWIndex(dimension: 4, metric: .dotProduct, params: params)

        // Generate 200 levels from each index — they must be identical.
        let levels1 = (0..<200).map { _ in index1.randomLevel() }
        let levels2 = (0..<200).map { _ in index2.randomLevel() }

        XCTAssertEqual(levels1, levels2,
            "Same seed must produce identical level sequence across two separate HNSWIndex instances")
    }

    func testRandomLevelDifferentSeedsDiffer() {
        var index1 = HNSWIndex(dimension: 4, metric: .dotProduct,
                               params: HNSWParameters(seed: 0xAAAA_BBBB))
        var index2 = HNSWIndex(dimension: 4, metric: .dotProduct,
                               params: HNSWParameters(seed: 0xCCCC_DDDD))

        let levels1 = (0..<50).map { _ in index1.randomLevel() }
        let levels2 = (0..<50).map { _ in index2.randomLevel() }

        XCTAssertNotEqual(levels1, levels2,
            "Different seeds must produce different level sequences (statistical near-certainty)")
    }

    /// Verifies that two HNSWIndex instances built from the same seed and same insert
    /// order have identical graph topology: for every node at every layer, the neighbor
    /// list is byte-identical (same IDs in the same order).
    func testGraphStructureDeterminismAcrossTwoRuns() throws {
        let dim    = 8
        let count  = 50
        let params = HNSWParameters(M: 4, efConstruction: 16, efSearch: 8, seed: 0x1234_5678)
        let vectors = TestFixtures.randomVectors(count: count, dimension: dim, seed: 0xCAFE_BABE)

        // Build run 1
        var index1 = HNSWIndex(dimension: dim, metric: .dotProduct, params: params)
        for (i, v) in vectors.enumerated() {
            try v.withUnsafeBufferPointer { buf in
                try index1.insert(internalID: Int32(i), vector: buf.baseAddress!)
            }
        }

        // Build run 2 — same params, same vectors, same insert order
        var index2 = HNSWIndex(dimension: dim, metric: .dotProduct, params: params)
        for (i, v) in vectors.enumerated() {
            try v.withUnsafeBufferPointer { buf in
                try index2.insert(internalID: Int32(i), vector: buf.baseAddress!)
            }
        }

        // Verify: every node's neighbor list at every layer must be identical.
        for id in 0..<Int32(count) {
            // Check both layer 0 and a couple of upper layers if they exist.
            for layer in 0..<5 {
                let n1 = index1.neighbors(of: id, at: layer)
                let n2 = index2.neighbors(of: id, at: layer)
                XCTAssertEqual(n1, n2,
                    "Node \(id) layer \(layer) neighbors differ between run 1 and run 2 (RNG not deterministic)")
            }
        }
    }
    
    // MARK: - Phase 9 Recall Correctness Tests
    
    private func runRecallTest(datasetSize: Int, dimension: Int = 16, k: Int = 10, targetRecall: Double = 0.90) throws {
        let vectors = TestFixtures.randomVectors(count: datasetSize, dimension: dimension, seed: 0x9876_5432)
        let queries = TestFixtures.randomVectors(count: 100, dimension: dimension, seed: 0x1111_2222)
        
        var flat = FlatIndex(dimension: dimension, metric: .dotProduct)
        var hnsw = HNSWIndex(dimension: dimension, metric: .dotProduct, params: HNSWParameters(M: 16, efConstruction: 200, efSearch: 50, seed: 42))
        
        for (i, v) in vectors.enumerated() {
            try v.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!
                try flat.insert(internalID: Int32(i), vector: ptr)
                try hnsw.insert(internalID: Int32(i), vector: ptr)
            }
        }
        
        var totalRecall: Double = 0.0
        
        for q in queries {
            let flatResults = q.withUnsafeBufferPointer { buf in
                flat.search(query: buf.baseAddress!, k: k, ef: nil)
            }
            let hnswResults = q.withUnsafeBufferPointer { buf in
                hnsw.search(query: buf.baseAddress!, k: k, ef: 50)
            }
            
            let flatIDs = Set(flatResults.map { $0.id })
            let hnswIDs = Set(hnswResults.map { $0.id })
            
            let intersection = flatIDs.intersection(hnswIDs)
            totalRecall += Double(intersection.count) / Double(k)
        }
        
        let averageRecall = totalRecall / Double(queries.count)
        print("Recall@\(k) for dataset size \(datasetSize): \(averageRecall)")
        XCTAssertGreaterThanOrEqual(averageRecall, targetRecall, "Average recall \(averageRecall) fell below threshold \(targetRecall) for dataset size \(datasetSize)")
    }
    
    func testRecallAt1k() throws {
        try runRecallTest(datasetSize: 1_000)
    }
    
    func testRecallAt10k() throws {
        try runRecallTest(datasetSize: 10_000)
    }
    
    func testRecallAt50k() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["RUN_SLOW_TESTS"] != nil, "Skipping slow 50k recall test. Set RUN_SLOW_TESTS=1 to run.")
        try runRecallTest(datasetSize: 50_000)
    }
}

// MARK: - Test helpers

/// Expose read-only neighbor list access for testing (not part of public API).
extension HNSWIndex {
    func neighbors(of internalID: Int32, at layer: Int) -> [Int32] {
        graphStorage.neighbors(of: internalID, at: layer)
    }
}
