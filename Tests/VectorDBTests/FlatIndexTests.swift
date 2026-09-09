/// FlatIndexTests.swift — Exact correctness tests for FlatIndex (Phase 1, §5).
///
/// Definition of Done coverage (§5):
///   [DoD-1] Insert 10k random vectors, verify top-k against independent
///           reference computation (the "numpy equivalent") for ≥5 queries,
///           across all three metrics.
///   [DoD-2] Empty-index and k=0 edge cases handled without crashing.
///   [DoD-3] Unit tests pass for all three metrics.

import XCTest
@testable import VectorDB

final class FlatIndexTests: XCTestCase {

    // MARK: - Helpers

    /// Build a FlatIndex from an array of [[Float]], assigning sequential Int32 IDs.
    private func buildIndex(
        vectors: [[Float]],
        dimension: Int,
        metric: DistanceMetric
    ) throws -> FlatIndex {
        var index = FlatIndex(dimension: dimension, metric: metric)
        for (i, v) in vectors.enumerated() {
            try TestFixtures.insert(into: &index, internalID: Int32(i), vector: v)
        }
        return index
    }

    // MARK: - [DoD-1a] .dotProduct: 10k vectors, 5 queries vs. reference

    func testDotProduct10kVectorsAgainstReference() throws {
        let dim = 32
        let count = 10_000
        let k = 10

        let vectors = TestFixtures.randomVectors(count: count, dimension: dim, seed: 0xDEAD_BEEF_0001)
        let index = try buildIndex(vectors: vectors, dimension: dim, metric: .dotProduct)

        XCTAssertEqual(index.count, count)

        let queries = TestFixtures.randomVectors(count: 5, dimension: dim, seed: 0xCAFE_BABE_0001)
        for (qi, query) in queries.enumerated() {
            let got    = TestFixtures.search(index: index, query: query, k: k)
            let ref    = TestFixtures.referenceDotProduct(query: query, vectors: vectors, k: k)

            XCTAssertEqual(got.count, k, "Query \(qi): expected \(k) results, got \(got.count)")

            // IDs must match the reference exactly (deterministic sort).
            let gotIDs = got.map { Int($0.id) }
            let refIDs = ref.map { $0.index }
            XCTAssertEqual(gotIDs, refIDs,
                "Query \(qi) .dotProduct: top-\(k) IDs differ from reference. " +
                "got=\(gotIDs) ref=\(refIDs)")

            // Scores must match within float rounding error.
            for (r, g) in zip(ref, got) {
                XCTAssertEqual(r.score, g.score, accuracy: 1e-5,
                    "Query \(qi): score mismatch at result for id \(g.id)")
            }
        }
    }

    // MARK: - [DoD-1b] .cosine: 10k unit-normalized vectors, 5 queries vs. reference

    func testCosine10kVectorsAgainstReference() throws {
        let dim = 32
        let count = 10_000
        let k = 10

        // Unit-normalized at generation — cosine = dot product.
        let vectors = TestFixtures.randomUnitVectors(count: count, dimension: dim, seed: 0xDEAD_BEEF_0002)
        let index = try buildIndex(vectors: vectors, dimension: dim, metric: .cosine)

        let queries = TestFixtures.randomUnitVectors(count: 5, dimension: dim, seed: 0xCAFE_BABE_0002)
        for (qi, query) in queries.enumerated() {
            let got = TestFixtures.search(index: index, query: query, k: k)
            // Reference: dot product on unit vectors == cosine similarity.
            let ref = TestFixtures.referenceDotProduct(query: query, vectors: vectors, k: k)

            let gotIDs = got.map { Int($0.id) }
            let refIDs = ref.map { $0.index }
            XCTAssertEqual(gotIDs, refIDs,
                "Query \(qi) .cosine: top-\(k) IDs differ from reference. " +
                "got=\(gotIDs) ref=\(refIDs)")

            for (r, g) in zip(ref, got) {
                XCTAssertEqual(r.score, g.score, accuracy: 1e-5,
                    "Query \(qi) .cosine: score mismatch for id \(g.id)")
            }
        }
    }

    // MARK: - [DoD-1c] .euclidean: 10k vectors, 5 queries vs. reference

    func testEuclidean10kVectorsAgainstReference() throws {
        let dim = 32
        let count = 10_000
        let k = 10

        let vectors = TestFixtures.randomVectors(count: count, dimension: dim, seed: 0xDEAD_BEEF_0003)
        let index = try buildIndex(vectors: vectors, dimension: dim, metric: .euclidean)

        let queries = TestFixtures.randomVectors(count: 5, dimension: dim, seed: 0xCAFE_BABE_0003)
        for (qi, query) in queries.enumerated() {
            let got = TestFixtures.search(index: index, query: query, k: k)
            let ref = TestFixtures.referenceEuclidean(query: query, vectors: vectors, k: k)

            let gotIDs = got.map { Int($0.id) }
            let refIDs = ref.map { $0.index }
            XCTAssertEqual(gotIDs, refIDs,
                "Query \(qi) .euclidean: top-\(k) IDs differ from reference. " +
                "got=\(gotIDs) ref=\(refIDs)")

            for (r, g) in zip(ref, got) {
                XCTAssertEqual(r.score, g.score, accuracy: 1e-5,
                    "Query \(qi) .euclidean: score mismatch for id \(g.id)")
            }
        }
    }

    // MARK: - [DoD-1d] Score correctness on tiny, hand-verifiable vectors

    func testHandVerifiableDotProductScores() throws {
        // 2-D vectors we can compute mentally:
        //   v0 = [1, 0], v1 = [0, 1], v2 = [0.6, 0.8]
        // query = [1, 0]
        //   dot(query, v0) = 1.0
        //   dot(query, v1) = 0.0
        //   dot(query, v2) = 0.6
        // Expected top-2: v0 (1.0), v2 (0.6)
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [0.0, 1.0])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [0.6, 0.8])

        let results = TestFixtures.search(index: index, query: [1.0, 0.0], k: 2)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].id, 0)
        XCTAssertEqual(results[0].score, 1.0, accuracy: 1e-6)
        XCTAssertEqual(results[1].id, 2)
        XCTAssertEqual(results[1].score, 0.6, accuracy: 1e-6)
    }

    func testHandVerifiableEuclideanScores() throws {
        // v0 = [0, 0], v1 = [3, 4], v2 = [1, 0]
        // query = [0, 0]
        //   -distSq(query, v0) =  0.0
        //   -distSq(query, v1) = -25.0
        //   -distSq(query, v2) = -1.0
        // Expected top-2: v0 (0.0), v2 (-1.0)
        var index = FlatIndex(dimension: 2, metric: .euclidean)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [0.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [3.0, 4.0])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [1.0, 0.0])

        let results = TestFixtures.search(index: index, query: [0.0, 0.0], k: 2)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].id, 0)
        XCTAssertEqual(results[0].score,  0.0, accuracy: 1e-6)
        XCTAssertEqual(results[1].id, 2)
        XCTAssertEqual(results[1].score, -1.0, accuracy: 1e-6)
    }

    // MARK: - [DoD-2] Edge cases — must not crash, must return []

    func testEmptyIndexReturnsEmpty() {
        let index = FlatIndex(dimension: 4, metric: .cosine)
        let results = TestFixtures.search(index: index, query: [1, 0, 0, 0], k: 5)
        XCTAssertTrue(results.isEmpty, "Search on empty index must return []")
    }

    func testKZeroReturnsEmpty() throws {
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        let results = TestFixtures.search(index: index, query: [1.0, 0.0], k: 0)
        XCTAssertTrue(results.isEmpty, "k=0 must return [] without throwing")
    }

    func testKGreaterThanCountReturnsAllLive() throws {
        // Insert 3 vectors, ask for k=100 — should return exactly 3 results.
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [0.0, 1.0])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [0.5, 0.5])

        let results = TestFixtures.search(index: index, query: [1.0, 0.0], k: 100)
        XCTAssertEqual(results.count, 3,
            "k > count must return all live vectors, not throw or crash")
    }

    func testKEqualToCount() throws {
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [0.0, 1.0])

        let results = TestFixtures.search(index: index, query: [0.5, 0.5], k: 2)
        XCTAssertEqual(results.count, 2)
    }

    // MARK: - Tombstone / remove edge cases

    func testRemoveFiltersTombstonedFromResults() throws {
        // v0=[1,0] scores highest vs. query=[1,0]; tombstone it; must not appear.
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [0.6, 0.8])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [0.0, 1.0])

        XCTAssertEqual(index.count, 3)
        try index.remove(internalID: 0)
        XCTAssertEqual(index.count, 2, "count must reflect tombstoned entry")

        let results = TestFixtures.search(index: index, query: [1.0, 0.0], k: 5)
        XCTAssertFalse(results.contains { $0.id == 0 },
            "Tombstoned id=0 must not appear in search results")
        XCTAssertEqual(results.count, 2)
        // id=1 (score 0.6) should now be top result
        XCTAssertEqual(results[0].id, 1)
    }

    func testRemoveNonExistentThrowsNotFound() {
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        XCTAssertThrowsError(try index.remove(internalID: 99)) { error in
            guard case VectorDBError.notFound = error else {
                XCTFail("Expected .notFound, got \(error)")
                return
            }
        }
    }

    func testRemoveAllThenSearchReturnsEmpty() throws {
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [0.0, 1.0])

        try index.remove(internalID: 0)
        try index.remove(internalID: 1)

        XCTAssertEqual(index.count, 0)
        let results = TestFixtures.search(index: index, query: [1.0, 0.0], k: 5)
        XCTAssertTrue(results.isEmpty,
            "Search on index with all entries tombstoned must return [], not crash")
    }

    // MARK: - [DoD-3] Sorting direction correctness across all metrics

    func testDescendingSortDotProduct() throws {
        // Insert 3 vectors with known ordering vs. query=[1,0,0]:
        //   v0=[1,0,0] → 1.0  (best)
        //   v1=[0.5,0,0] → 0.5
        //   v2=[-1,0,0] → -1.0 (worst)
        var index = FlatIndex(dimension: 3, metric: .dotProduct)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [ 1.0, 0.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [ 0.5, 0.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [-1.0, 0.0, 0.0])

        let results = TestFixtures.search(index: index, query: [1.0, 0.0, 0.0], k: 3)
        XCTAssertEqual(results.map { $0.id }, [0, 1, 2],
            ".dotProduct results must be sorted highest-score-first")
        XCTAssertTrue(results[0].score > results[1].score &&
                      results[1].score > results[2].score,
            "Scores must be strictly descending")
    }

    func testDescendingSortEuclidean() throws {
        // query=[0,0]: closest is v0=[0,0] (distSq=0), then v1=[1,0] (distSq=1), then v2=[3,4] (distSq=25).
        // Score = -distSq → v0(0), v1(-1), v2(-25)  ← descending is correct.
        var index = FlatIndex(dimension: 2, metric: .euclidean)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [0.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [1.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [3.0, 4.0])

        let results = TestFixtures.search(index: index, query: [0.0, 0.0], k: 3)
        XCTAssertEqual(results.map { $0.id }, [0, 1, 2],
            ".euclidean results must be sorted closest-first (highest -distSq)")
        XCTAssertEqual(results[0].score,   0.0, accuracy: 1e-6)
        XCTAssertEqual(results[1].score,  -1.0, accuracy: 1e-6)
        XCTAssertEqual(results[2].score, -25.0, accuracy: 1e-6)
    }

    func testDescendingSortCosine() throws {
        // Unit vectors: [1,0], [√2/2, √2/2], [0,1]. Query=[1,0].
        // cosine (= dot) scores: 1.0, ~0.707, 0.0 → expected order: v0, v1, v2.
        let sq2: Float = 0.70710678
        var index = FlatIndex(dimension: 2, metric: .cosine)
        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        try TestFixtures.insert(into: &index, internalID: 1, vector: [sq2,  sq2])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [0.0, 1.0])

        let results = TestFixtures.search(index: index, query: [1.0, 0.0], k: 3)
        XCTAssertEqual(results.map { $0.id }, [0, 1, 2],
            ".cosine results must be sorted highest-similarity-first")
        XCTAssertEqual(results[0].score, 1.0,  accuracy: 1e-6)
        XCTAssertEqual(results[1].score, sq2,  accuracy: 1e-5)
        XCTAssertEqual(results[2].score, 0.0,  accuracy: 1e-6)
    }

    // MARK: - Count consistency

    func testCountAfterInsertsAndTombstones() throws {
        var index = FlatIndex(dimension: 2, metric: .dotProduct)
        XCTAssertEqual(index.count, 0)

        try TestFixtures.insert(into: &index, internalID: 0, vector: [1.0, 0.0])
        XCTAssertEqual(index.count, 1)

        try TestFixtures.insert(into: &index, internalID: 1, vector: [0.0, 1.0])
        try TestFixtures.insert(into: &index, internalID: 2, vector: [0.5, 0.5])
        XCTAssertEqual(index.count, 3)

        try index.remove(internalID: 1)
        XCTAssertEqual(index.count, 2, "count must decrement after tombstone")

        try index.remove(internalID: 0)
        XCTAssertEqual(index.count, 1)
    }
}

// MARK: - Phase 2 VectorMath Accelerate tests (§6)

/// VectorMathAccelerateTests — Phase 2 Definition of Done correctness tests.
///
/// Definition of Done coverage (§6):
///   [DoD-A] Cosine-via-normalization matches textbook dot(a,b)/(normA*normB).
///   [DoD-B] Zero-vector: normalize() is a no-op; isNonZeroVector() detects it.
///   [DoD-C] NaN/Inf inputs detected by isFiniteVector().
///   [DoD-D] batchDot() results match per-vector dot() calls.
///   [DoD-E] Phase 1 FlatIndex top-k results are bit-equivalent after Phase 2 swap.
final class VectorMathAccelerateTests: XCTestCase {

    // MARK: [DoD-A] cosine-via-normalization == textbook dot/(normA*normB)

    func testCosineViaNormalizationMatchesTextbook() {
        var a: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
        var b: [Float] = [2, 7, 1, 8, 2, 8, 1, 8]
        let rawDot   = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let normA    = sqrt(a.reduce(Float(0)) { $0 + $1 * $1 })
        let normB    = sqrt(b.reduce(Float(0)) { $0 + $1 * $1 })
        let textbook = rawDot / (normA * normB)
        a.withUnsafeMutableBufferPointer { VectorMath.normalize($0.baseAddress!, $0.count) }
        b.withUnsafeMutableBufferPointer { VectorMath.normalize($0.baseAddress!, $0.count) }
        let vdb = a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                VectorMath.similarity(ap.baseAddress!, bp.baseAddress!, a.count, metric: .cosine)
            }
        }
        XCTAssertEqual(vdb, textbook, accuracy: 1e-5,
            "cosine-via-normalization must match dot/(normA*normB): got \(vdb), expected \(textbook)")
    }

    func testNormalizedVectorHasUnitLength() {
        var v: [Float] = [3, 4]
        v.withUnsafeMutableBufferPointer { VectorMath.normalize($0.baseAddress!, $0.count) }
        let norm = sqrt(v.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-6)
        XCTAssertEqual(v[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(v[1], 0.8, accuracy: 1e-6)
    }

    // MARK: [DoD-B] Zero-vector

    func testNormalizeZeroVectorIsNoOp() {
        var v = [Float](repeating: 0.0, count: 8)
        v.withUnsafeMutableBufferPointer { VectorMath.normalize($0.baseAddress!, $0.count) }
        for element in v { XCTAssertEqual(element, 0.0) }
    }

    func testIsNonZeroVectorFalseForZero() {
        let zero = [Float](repeating: 0.0, count: 16)
        XCTAssertFalse(zero.withUnsafeBufferPointer {
            VectorMath.isNonZeroVector($0.baseAddress!, $0.count)
        })
    }

    func testIsNonZeroVectorTrueForNonZero() {
        let v: [Float] = [0, 0, 0, 1, 0, 0, 0, 0]
        XCTAssertTrue(v.withUnsafeBufferPointer {
            VectorMath.isNonZeroVector($0.baseAddress!, $0.count)
        })
    }

    func testNearZeroTreatedAsZeroAndNoNaN() {
        // sumSq = 4*(1e-7)^2 = 4e-14 < 1e-12 threshold
        var v: [Float] = [1e-7, 1e-7, 1e-7, 1e-7]
        XCTAssertFalse(v.withUnsafeBufferPointer {
            VectorMath.isNonZeroVector($0.baseAddress!, $0.count)
        }, "sumSq below 1e-12 must be treated as zero")
        v.withUnsafeMutableBufferPointer { VectorMath.normalize($0.baseAddress!, $0.count) }
        for e in v { XCTAssertTrue(e.isFinite, "normalize() on near-zero must not produce NaN/Inf") }
    }

    // MARK: [DoD-C] NaN/Inf detection

    func testIsFiniteReturnsFalseForNaN() {
        let v: [Float] = [1.0, .nan, 3.0]
        XCTAssertFalse(v.withUnsafeBufferPointer {
            VectorMath.isFiniteVector($0.baseAddress!, $0.count)
        })
    }

    func testIsFiniteReturnsFalseForInf() {
        let v: [Float] = [1.0, .infinity, 3.0]
        XCTAssertFalse(v.withUnsafeBufferPointer {
            VectorMath.isFiniteVector($0.baseAddress!, $0.count)
        })
    }

    func testIsFiniteReturnsTrueForValidVector() {
        let v: [Float] = [1.0, -2.5, 0.0, 3.14]
        XCTAssertTrue(v.withUnsafeBufferPointer {
            VectorMath.isFiniteVector($0.baseAddress!, $0.count)
        })
    }

    // MARK: [DoD-D] batchDot vs. per-vector dot()

    func testBatchDotMatchesPerVectorDot() {
        let dim = 32, n = 50
        let vectors = TestFixtures.randomVectors(count: n, dimension: dim, seed: 0xBA7C_0001)
        let query   = TestFixtures.randomVectors(count: 1, dimension: dim, seed: 0xBA7C_CAFE).first!
        let flat    = vectors.flatMap { $0 }
        let batch   = flat.withUnsafeBufferPointer { fp in
            query.withUnsafeBufferPointer { qp in
                VectorMath.batchDot(query: qp.baseAddress!, vectors: fp.baseAddress!, count: n, dim: dim)
            }
        }
        for (i, v) in vectors.enumerated() {
            let perVec = query.withUnsafeBufferPointer { qp in
                v.withUnsafeBufferPointer { vp in VectorMath.dot(qp.baseAddress!, vp.baseAddress!, dim) }
            }
            XCTAssertEqual(batch[i], perVec, accuracy: 1e-4, "batchDot[\(i)] must match dot()")
        }
    }

    // MARK: [DoD-E] Phase 1 regression

    func testFlatIndexResultsUnchangedAfterAccelerateSwap() throws {
        let dim = 32, count = 10_000, k = 10
        let vectors = TestFixtures.randomVectors(count: count, dimension: dim, seed: 0xDEAD_BEEF_0001)
        var index = FlatIndex(dimension: dim, metric: .dotProduct)
        for (i, v) in vectors.enumerated() {
            try TestFixtures.insert(into: &index, internalID: Int32(i), vector: v)
        }
        let query = TestFixtures.randomVectors(count: 1, dimension: dim, seed: 0xCAFE_BABE_0001).first!
        let got   = TestFixtures.search(index: index, query: query, k: k)
        let ref   = TestFixtures.referenceDotProduct(query: query, vectors: vectors, k: k)
        XCTAssertEqual(got.map { Int($0.id) }, ref.map { $0.index },
            "Phase 2 swap must not change FlatIndex top-k")
    }
}
