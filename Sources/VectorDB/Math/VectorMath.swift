/// VectorMath.swift — Phase 2: Accelerate/vDSP implementation.
///
/// SIGNATURE CONTRACT: All function signatures are IDENTICAL to the Phase 1
/// scalar version. FlatIndex, HNSWIndex, and every test file that calls
/// VectorMath need zero changes when Phase 1 is replaced by this file.
///
/// Score convention (unchanged from Phase 1):
///   Higher score = more similar, for ALL three metrics.
///   .euclidean returns -distanceSquared so the sort direction is uniform.
///
/// Pitfalls addressed (§6):
///   • Zero vectors:    normalize() silently skips if sumSq ≤ 1e-12.
///                      isNonZeroVector() lets callers detect this before insert.
///   • NaN/Inf inputs:  isFiniteVector() scans all elements; call at public API
///                      layer before any unsafe pointer code is reached (Phase 8).
///   • Memory alignment: vDSP on Apple Silicon does NOT require hand-rolled
///                       alignment. UnsafeMutablePointer<Float>.allocate()
///                       already provides suitably aligned memory — no manual
///                       alignment logic anywhere in this file.
///   • Dimension mismatch: validated at the public API layer (VectorDB actor,
///                          Phase 8) before calling any function here. Passing
///                          a mismatched dim to an unsafe-pointer function is a
///                          buffer overrun, not a Swift-safe trap.

import Accelerate

enum VectorMath {

    // MARK: - Primary similarity entry point (signature identical to Phase 1)

    /// Compute the similarity score between two vectors of length `dim`.
    ///
    /// Score convention: **higher always means more similar**, for all metrics.
    ///
    /// - Note: For `.cosine`, the caller must pre-normalize both vectors via
    ///   `normalize(_:_:)` at insert time. Cosine similarity of unit vectors
    ///   degenerates to a plain dot product, which is what this function
    ///   computes — avoiding an extra divide per query.
    static func similarity(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        _ dim: Int,
        metric: DistanceMetric
    ) -> Float {
        switch metric {
        case .dotProduct:
            return dot(a, b, dim)
        case .cosine:
            // Pre-normalization assumed. See normalize() below.
            // If you cannot guarantee unit-length input, the full formula is:
            //   dot(a,b,dim) / (sqrt(dot(a,a,dim)) * sqrt(dot(b,b,dim)))
            // but that's never needed in this architecture since VectorDB
            // (Phase 8) normalizes every vector exactly once on insert.
            return dot(a, b, dim)
        case .euclidean:
            var distSq: Float = 0
            vDSP_distancesq(a, 1, b, 1, &distSq, vDSP_Length(dim))
            return -distSq   // negate: lower distance → higher score
        }
    }

    // MARK: - Dot product

    /// Accelerate-backed dot product of two float vectors.
    /// Uses vDSP_dotpr: SIMD-vectorised, no Swift Array overhead.
    static func dot(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        _ dim: Int
    ) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(dim))
        return result
    }

    // MARK: - In-place L2 normalization

    /// Normalizes `v` to unit length in place. Call once at insert time for
    /// every vector stored under the `.cosine` metric.
    ///
    /// If `sumSq ≤ 1e-12` (zero or near-zero vector), the function returns
    /// without modifying `v`. The caller (VectorDB public actor, Phase 8) is
    /// responsible for detecting and rejecting zero vectors via
    /// `isNonZeroVector(_:_:)` before calling this, so that the insert
    /// throws `.invalidVector` rather than silently storing a zero vector
    /// whose cosine similarity to everything is undefined.
    static func normalize(_ v: UnsafeMutablePointer<Float>, _ dim: Int) {
        var sumSq: Float = 0
        vDSP_svesq(v, 1, &sumSq, vDSP_Length(dim))
        guard sumSq > 1e-12 else { return }
        var scale = 1.0 / sqrtf(sumSq)
        vDSP_vsmul(v, 1, &scale, v, 1, vDSP_Length(dim))
    }

    // MARK: - Batched dot product (sgemv)

    /// Compute the dot product of a single `query` against `count` stored
    /// vectors packed row-major in `vectors` (shape: count × dim).
    ///
    /// Uses `cblas_sgemv` (matrix-vector multiply), which is cache-blocked and
    /// takes full advantage of SIMD width across the entire batch — far faster
    /// than calling `dot()` in a loop for large `count`.
    ///
    /// Use case: flat re-ranking of HNSW approximate candidates (Phase 4),
    /// or the brute-force fallback path for small collections.
    ///
    /// - Note: Reach for `cblas_sgemm` (matrix-matrix) only when you have
    ///   *multiple simultaneous queries* (batch embedding). The single-query
    ///   hot path uses sgemv.
    static func batchDot(
        query: UnsafePointer<Float>,
        vectors: UnsafePointer<Float>,
        count: Int,
        dim: Int
    ) -> [Float] {
        var results = [Float](repeating: 0, count: count)
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            Int32(count), Int32(dim),
            1.0, vectors, Int32(dim),
            query, 1,
            0.0, &results, 1
        )
        return results
    }

    // MARK: - Validation helpers (called at public API layer, Phase 8)

    /// Returns `true` if every element of `v` is a finite Float (no NaN, no Inf).
    /// Scan cost is O(dim) — cheap relative to corrupted-graph consequences.
    static func isFiniteVector(_ v: UnsafePointer<Float>, _ dim: Int) -> Bool {
        for i in 0..<dim where !v[i].isFinite { return false }
        return true
    }

    /// Returns `true` if the vector's squared norm exceeds 1e-12.
    /// A false return means normalize() will be a no-op and cosine
    /// similarity to anything will be undefined (all zeros).
    static func isNonZeroVector(_ v: UnsafePointer<Float>, _ dim: Int) -> Bool {
        var sumSq: Float = 0
        vDSP_svesq(v, 1, &sumSq, vDSP_Length(dim))
        return sumSq > 1e-12
    }
}
