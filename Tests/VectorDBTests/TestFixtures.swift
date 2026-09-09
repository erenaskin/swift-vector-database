/// TestFixtures.swift — Deterministic seeded vector generation and reference computations.
///
/// ALL test data MUST be generated through these fixtures with fixed seeds.
/// Rationale: "does my recall regression test still pass" is meaningless if the
/// data changes randomly between runs (§8.4). This same principle applies to
/// FlatIndex correctness tests.
///
/// The reference brute-force functions are intentionally INDEPENDENT from
/// FlatIndex — they use only Swift standard library reduce/zip so they serve
/// as the "numpy equivalent" verification source described in §5's Definition
/// of Done. If FlatIndex and the reference disagree, FlatIndex has a bug.

import XCTest
@testable import VectorDB




// MARK: - TestFixtures

enum TestFixtures {

    // MARK: Vector generation

    /// Generates `count` random vectors in [-1, 1]^dim with a fixed seed.
    /// NOT normalized. Use for .dotProduct and .euclidean tests.
    static func randomVectors(
        count: Int,
        dimension: Int,
        seed: UInt64 = 0xDEAD_BEEF_CAFE_1234
    ) -> [[Float]] {
        var rng = SeedableRNG(seed: seed)
        return (0..<count).map { _ in
            (0..<dimension).map { _ in rng.nextFloat() }
        }
    }

    /// Generates `count` L2-unit-normalized random vectors with a fixed seed.
    /// For .cosine tests: cosine similarity of unit vectors = dot product,
    /// so no separate cosine formula is needed in the reference.
    static func randomUnitVectors(
        count: Int,
        dimension: Int,
        seed: UInt64 = 0xABCD_EF01_2345_6789
    ) -> [[Float]] {
        let raw = randomVectors(count: count, dimension: dimension, seed: seed)
        return raw.map { v in
            let norm = sqrt(v.reduce(Float(0)) { $0 + $1 * $1 })
            guard norm > 1e-6 else {
                // Near-zero vector: replace with unit vector along axis 0.
                // This cannot happen in practice with a good seed, but guards
                // against silent NaN propagation if it did.
                var unit = [Float](repeating: 0, count: v.count)
                if !unit.isEmpty { unit[0] = 1.0 }
                return unit
            }
            return v.map { $0 / norm }
        }
    }

    // MARK: Reference brute-force (the "numpy equivalent")
    // These functions are the ground truth — they must NOT call any VectorDB code.

    /// Top-k by dot product, descending. Reference for .dotProduct and .cosine.
    static func referenceDotProduct(
        query: [Float],
        vectors: [[Float]],
        k: Int
    ) -> [(index: Int, score: Float)] {
        guard k > 0, !vectors.isEmpty else { return [] }
        let scores: [(Int, Float)] = vectors.enumerated().map { i, v in
            let dot = zip(query, v).reduce(Float(0)) { acc, pair in acc + pair.0 * pair.1 }
            return (i, dot)
        }
        return Array(scores.sorted { $0.1 > $1.1 }.prefix(k))
            .map { (index: $0.0, score: $0.1) }
    }

    /// Top-k by euclidean (score = -distSq), descending. Reference for .euclidean.
    static func referenceEuclidean(
        query: [Float],
        vectors: [[Float]],
        k: Int
    ) -> [(index: Int, score: Float)] {
        guard k > 0, !vectors.isEmpty else { return [] }
        let scores: [(Int, Float)] = vectors.enumerated().map { i, v in
            let distSq = zip(query, v).reduce(Float(0)) { acc, pair in
                let d = pair.0 - pair.1; return acc + d * d
            }
            return (i, -distSq)
        }
        return Array(scores.sorted { $0.1 > $1.1 }.prefix(k))
            .map { (index: $0.0, score: $0.1) }
    }

    // MARK: Convenience call helpers

    /// Inserts `vector` into `index` using the safe withUnsafeBufferPointer pattern.
    static func insert(
        into index: inout FlatIndex,
        internalID: Int32,
        vector: [Float]
    ) throws {
        try vector.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else {
                throw VectorDBError.invalidVector(reason: "empty vector array")
            }
            try index.insert(internalID: internalID, vector: base)
        }
    }

    /// Runs `search` on `index` using the safe withUnsafeBufferPointer pattern.
    static func search(
        index: FlatIndex,
        query: [Float],
        k: Int,
        ef: Int? = nil
    ) -> [(id: Int32, score: Float)] {
        query.withUnsafeBufferPointer { buf in
            // query is never empty in any test — buf.baseAddress! is safe here.
            index.search(query: buf.baseAddress!, k: k, ef: ef)
        }
    }
}
