/// SeedableRNG.swift — Deterministic xorshift64 pseudo-random number generator.
///
/// Used by `HNSWIndex.randomLevel()` for reproducible graph construction (§8.4).
///
/// Rationale: HNSW level assignment MUST be deterministic for reproducible recall
/// regression tests. The default `Double.random` is seeded by the OS at launch, so
/// "does my recall test still pass" is meaningless if the graph shape changes every run.
///
/// xorshift64 properties:
///   - Period: 2^64 − 1 (every non-zero state is visited exactly once before cycling).
///   - Fixed point: state=0 is the only cycle of length 1. Seed must never be 0.
///   - Conforms to Swift's `RandomNumberGenerator` so it can be passed to
///     `Double.random(in:using:)` directly.
public struct SeedableRNG: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) {
        precondition(seed != 0, "SeedableRNG: seed must not be 0 (xorshift64 fixed point)")
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        var x = state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        state = x
        return x
    }

    /// Returns a Float uniformly in [-1, 1].
    public mutating func nextFloat() -> Float {
        let bits = next()
        // Take 24 mantissa bits, map to [0, 1), then shift to [-1, 1).
        let magnitude = Float(bits >> 40) / Float(1 << 24)
        return (bits & (1 << 39) != 0) ? -magnitude : magnitude
    }
}
