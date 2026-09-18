/// VectorMath.swift — Accelerate/vDSP backed vector math.
///
/// Score convention:
///   Higher score = more similar, for ALL three metrics.
///   .euclidean returns -distanceSquared so the sort direction is uniform.
///
/// Pitfalls addressed (§6):
///   • Zero vectors:    normalize() silently skips if sumSq ≤ 1e-12.
///                      isNonZeroVector() lets callers detect this before insert.
///   • NaN/Inf inputs:  isFiniteVector() scans all elements; called at the public
///                      API layer (VectorDatabase.validate) before any unsafe pointer
///                      code is reached.
///   • Memory alignment: vDSP on Apple Silicon does NOT require hand-rolled
///                       alignment. UnsafeMutablePointer<Float>.allocate()
///                       already provides suitably aligned memory.
///   • Dimension mismatch: validated at the public API layer before calling any
///                         function here. Passing a mismatched dim to an
///                         unsafe-pointer function is a buffer overrun, not a
///                         Swift-safe trap.

import Accelerate

enum VectorMath {

    // MARK: - Primary similarity entry point

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
            return dot(a, b, dim)
        case .euclidean:
            var distSq: Float = 0
            vDSP_distancesq(a, 1, b, 1, &distSq, vDSP_Length(dim))
            return -distSq  // negate: lower distance → higher score
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

    /// Squared L2 norm (‖v‖²) of a single vector.
    ///
    /// Used by `FlatIndex` to cache one scalar per stored vector, which is what
    /// makes the genuinely-batched euclidean path below possible.
    static func squaredNorm(_ v: UnsafePointer<Float>, _ dim: Int) -> Float {
        var sumSq: Float = 0
        vDSP_svesq(v, 1, &sumSq, vDSP_Length(dim))
        return sumSq
    }

    // MARK: - In-place L2 normalization

    /// Normalizes `v` to unit length in place. Call once at insert time for
    /// every vector stored under the `.cosine` metric.
    ///
    /// If `sumSq ≤ 1e-12` (zero or near-zero vector), the function returns
    /// without modifying `v`. The caller (VectorDatabase public actor) is responsible
    /// for detecting and rejecting zero vectors via `isNonZeroVector(_:_:)`
    /// before calling this, so that the insert throws `.invalidVector` rather
    /// than silently storing a zero vector whose cosine similarity to everything
    /// is undefined.
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

    // MARK: - Batched Cosine

    /// Compute the cosine similarity of a single `query` against `count` stored
    /// vectors packed row-major in `vectors` (shape: count × dim).
    ///
    /// - Note: For `.cosine`, the caller must pre-normalize both vectors via
    ///   `normalize(_:_:)` at insert time. Cosine similarity of unit vectors
    ///   degenerates to a plain dot product, which is what this function
    ///   computes by delegating to `batchDot`.
    static func batchCosine(
        query: UnsafePointer<Float>,
        vectors: UnsafePointer<Float>,
        count: Int,
        dim: Int
    ) -> [Float] {
        return batchDot(query: query, vectors: vectors, count: count, dim: dim)
    }

    // MARK: - Batched Euclidean

    /// Compute `-‖query - vᵢ‖²` for every stored vector, using ONE `sgemv` call
    /// plus O(count) scalar work — instead of `count` separate `vDSP_distancesq`
    /// calls.
    ///
    /// FIX O4 — WHY THIS IS NOW A REAL BATCH:
    /// The previous implementation was a per-row `vDSP_distancesq` loop with a
    /// batch-shaped signature (its own comment admitted as much). Every iteration
    /// paid function-call overhead and re-streamed one row at a time, so
    /// `.euclidean` was measurably slower than `.dotProduct`/`.cosine` for no
    /// algorithmic reason.
    ///
    /// The identity used is the standard one:
    ///
    ///     ‖q - v‖² = ‖q‖² - 2·(q·v) + ‖v‖²
    ///     score    = -‖q - v‖² = 2·(q·v) - ‖q‖² - ‖v‖²
    ///
    /// `q·v` for every row comes from the same cache-blocked `sgemv` the dot
    /// product path already uses, and `‖v‖²` is read from `squaredNorms`, which
    /// `FlatIndex` maintains incrementally (one `vDSP_svesq` per insert, one
    /// array copy per swap-remove). `‖q‖²` is a single scalar that is identical
    /// for every row, so it does not affect ranking at all — only the reported
    /// score value.
    ///
    /// NUMERICAL NOTE: this expanded form involves a subtraction of similarly
    /// sized terms, so scores carry slightly more float32 rounding error than a
    /// direct `vDSP_distancesq`. For typical embedding magnitudes the relative
    /// error stays in the 1e-6 range, which is far below any meaningful
    /// similarity threshold, but it is a real trade-off and is documented here
    /// rather than hidden.
    ///
    /// - Parameter squaredNorms: `‖vᵢ‖²` for each stored row, in slot order.
    ///   Must contain at least `count` elements.
    static func batchEuclideanSquared(
        query: UnsafePointer<Float>,
        vectors: UnsafePointer<Float>,
        count: Int,
        dim: Int,
        squaredNorms: UnsafePointer<Float>
    ) -> [Float] {
        guard count > 0 else { return [] }

        // results = q · vᵢ  for every row, in one blocked matrix-vector multiply.
        var results = batchDot(query: query, vectors: vectors, count: count, dim: dim)

        let n = vDSP_Length(count)
        var two: Float = 2.0
        var negQueryNorm: Float = -squaredNorm(query, dim)

        results.withUnsafeMutableBufferPointer { buf in
            guard let out = buf.baseAddress else { return }
            // results = 2 · (q·vᵢ)
            vDSP_vsmul(out, 1, &two, out, 1, n)
            // vDSP_vsub(A, .., B, .., C, ..) computes C = B - A.
            // results = results - ‖vᵢ‖²
            vDSP_vsub(squaredNorms, 1, out, 1, out, 1, n)
            // results = results - ‖q‖²   (a constant offset; ranking is unaffected)
            vDSP_vsadd(out, 1, &negQueryNorm, out, 1, n)
        }

        return results
    }

    // MARK: - Validation helpers (called at the public API layer)

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
        return squaredNorm(v, dim) > 1e-12
    }
}
