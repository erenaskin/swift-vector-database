/// HNSWParameters.swift — Tunable hyperparameters for the HNSW index (§8.2).
///
/// These parameters dominate the recall / speed / memory tradeoff for the graph.
/// Default values (M=16, efConstruction=200, efSearch=50) are chosen for on-device
/// text-embedding workloads (hundreds to low tens of thousands of vectors) favoring
/// build/insert speed over marginal recall gains at web-scale.
import Foundation

public struct HNSWParameters: Sendable {

    /// Max neighbors per node at layer 1 and above.
    /// Higher = better recall, more memory, slower insert.
    ///
    /// FIX Y4: this and the other stored properties are `let`, not `var`. When
    /// they were mutable, a caller could construct a valid value and then assign
    /// `params.M = 1` afterwards, silently bypassing validation and producing a
    /// degenerate graph. Immutability makes "validated once" actually mean
    /// "validated forever".
    public let M: Int

    /// Candidate list size during insertion.
    /// Higher = better graph quality, much slower build.
    public let efConstruction: Int

    /// Candidate list size during query.
    /// Higher = better recall, slower query. Runtime-tunable knob via `search(ef:)`.
    public let efSearch: Int

    /// Seed for random level assignment (deterministic graph construction).
    ///
    /// MUST be non-zero: `HNSWIndex.init` hands this straight to
    /// `SeedableRNG(seed:)`, whose xorshift64 generator treats 0 as its only
    /// fixed point (it would get stuck emitting zeros forever). See the
    /// `seed != 0` check in `validate()` below for why that can no longer reach
    /// `SeedableRNG` unchecked.
    public let seed: UInt64

    /// Max neighbors per node at layer 0 (M * 2).
    /// Layer 0 needs denser connectivity since it's the final search layer.
    public var Mmax0: Int { M * 2 }

    /// Level-generation normalization factor (1 / ln(M)).
    /// Controls how many nodes end up on higher layers.
    public var mL: Double { 1.0 / log(Double(M)) }

    /// Creates a parameter set. This initializer never traps and never throws.
    ///
    /// FIX Y4 — WHY THIS NO LONGER CALLS `fatalError`:
    /// The previous version called `fatalError` for out-of-range values. A
    /// library has no business killing its host application because a number
    /// typed into a settings screen was wrong — especially a library whose whole
    /// public surface is already `throws`. Validation now happens through
    /// `validate()`, which `VectorDatabase.init` calls, so a bad configuration
    /// surfaces as a normal, catchable `VectorDatabaseError.invalidParameters`.
    ///
    /// Constructing an invalid value is therefore allowed; USING it through the
    /// public API is not. Call `validate()` yourself if you want to check a
    /// value before handing it to `VectorDatabase`.
    public init(M: Int = 16, efConstruction: Int = 200, efSearch: Int = 50, seed: UInt64 = 0x5EED) {
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.seed = seed
    }

    /// Throws `VectorDatabaseError.invalidParameters` if any value would produce a
    /// degenerate or unusable index.
    ///
    /// - `M` must be greater than 1: `M <= 1` makes the level-generation factor
    ///   `1 / ln(M)` zero, negative, or infinite, which silently produces a
    ///   degenerate graph (either everything piles onto layer 0, or `mL` is `+inf`).
    /// - `efConstruction` / `efSearch` must be positive: a non-positive candidate
    ///   list size cannot return any results.
    /// - `seed` must be non-zero.
    ///
    ///   FIX (this pass): this check was missing entirely, which meant a
    ///   `seed: 0` value sailed straight through `validate()` and only failed
    ///   much later, and far away from the call site that caused it —
    ///   `SeedableRNG.init` traps with `precondition(seed != 0, ...)` the
    ///   moment an `HNSWIndex` actually gets constructed with these params.
    ///   Because `IndexRouter` starts every fresh `VectorDatabase` on `FlatIndex` and
    ///   only builds the `HNSWIndex` once `hnswThreshold` (default 2,000) live
    ///   vectors have been inserted, this crash didn't happen at `VectorDatabase.init`
    ///   time at all: `VectorDatabase(dimension:parameters: .init(seed: 0))` would
    ///   succeed, and thousands of successful `insert()` calls could follow it —
    ///   the process would then die on whichever insert happens to cross the
    ///   threshold (or immediately, if constructing a `VectorDatabase` from a
    ///   persisted file/snapshot whose HNSW index is loaded up front). That is
    ///   exactly the "library kills its host app over a bad config value"
    ///   failure mode Fix Y4 (above) already removed for `M`/`efConstruction`/
    ///   `efSearch` — `seed` was simply the one value that check forgot.
    ///   Rejecting it here turns that delayed, confusing fatal crash back into
    ///   an ordinary, catchable `VectorDatabaseError.invalidParameters` right at
    ///   `VectorDatabase.init`, consistent with every other parameter.
    public func validate() throws {
        guard M > 1 else {
            throw VectorDatabaseError.invalidParameters(
                reason: "HNSWParameters.M must be greater than 1 (got \(M)).")
        }
        guard efConstruction > 0 else {
            throw VectorDatabaseError.invalidParameters(
                reason: "HNSWParameters.efConstruction must be positive (got \(efConstruction)).")
        }
        guard efSearch > 0 else {
            throw VectorDatabaseError.invalidParameters(
                reason: "HNSWParameters.efSearch must be positive (got \(efSearch)).")
        }
        guard seed != 0 else {
            throw VectorDatabaseError.invalidParameters(
                reason: "HNSWParameters.seed must not be 0 (0 is xorshift64's fixed point, "
                    + "which would leave SeedableRNG stuck producing zeros forever). "
                    + "Pass any non-zero UInt64, e.g. the default 0x5EED.")
        }
    }

    /// On-device recommendation defaults.
    public static let `default` = HNSWParameters()
}
