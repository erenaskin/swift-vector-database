/// VisitedList.swift — O(1)-reset epoch-based visited node tracker for HNSW graph traversal.
///
/// DESIGN RATIONALE:
///   `searchLayer` needs to track which nodes have been visited during graph traversal.
///   The naive approach (Swift's `Set<Int32>`) requires O(1) amortized inserts and lookups,
///   but the hashing overhead is severe in practice: the Instruments trace showed `Set.contains`,
///   `_rawHashValue`, and `Set.insert` collectively consuming ~13% of total CPU time during
///   500k-vector insertion benchmarks.
///
///   This implementation uses a dense `[UInt16]` array (one slot per possible node ID) and
///   an epoch counter. "Was this node visited?" reduces to a single integer comparison with
///   no hashing. Resetting the visited state between `searchLayer` calls costs O(1) —
///   just `currentEpoch += 1` — rather than the O(capacity) `memset` a BitSet would require.
///
/// OVERFLOW SAFETY:
///   When `currentEpoch` reaches `UInt16.max`, the next `nextEpoch()` call zeroes the
///   underlying array and resets `currentEpoch` to 1. This mirrors the strategy used in
///   hnswlib's `VisitedListPool` and prevents stale epoch values from causing false
///   "already visited" results — a silent recall-corruption bug, not a crash.
///
/// THREAD-SAFETY SCOPE:
///   `VisitedList` is intentionally allocated ONCE PER `insert` or `search` call and
///   passed `inout` to `searchLayer`. Do NOT lift it to an instance variable on `HNSWIndex`.
///   Doing so would break lock-free concurrent reads: two simultaneous `search` calls would
///   share and corrupt each other's epoch state. The current per-call allocation cost is
///   one `malloc` of ~capacity * 2 bytes total, amortised across all layers of a single
///   insert/search — negligible compared to the Set overhead it replaces.
///
/// EPOCH=0 INVARIANT:
///   The `epochs` array is zero-initialised by Swift. `currentEpoch` starts at 1.
///   Therefore a fresh `VisitedList` correctly reports `contains(_:) == false` for all
///   IDs before any `nextEpoch()` call — `0 != 1`.

struct VisitedList {

    // MARK: - Storage

    /// Dense epoch stamps, one per possible node ID.
    private var epochs: [UInt16]

    /// The stamp value that marks a node as "visited in the current traversal".
    /// Starts at 1 so that zero-initialised slots are never falsely "visited".
    private var currentEpoch: UInt16 = 1

    // MARK: - Init

    /// Allocates a `VisitedList` large enough for node IDs `0..<capacity`.
    ///
    /// - Parameter capacity: Must be >= the largest node ID that will be passed
    ///   to `insert` or `contains`. In `HNSWIndex`, use `nodeSlots.count` captured
    ///   **after** the new node has been appended to `nodeSlots`.
    init(capacity: Int) {
        // Swift zero-initialises [UInt16]. Combined with currentEpoch starting at 1,
        // this guarantees no node is falsely "visited" on the first traversal.
        epochs = [UInt16](repeating: 0, count: capacity)
    }

    // MARK: - API

    /// Marks the beginning of a new traversal (e.g. a new `searchLayer` call).
    ///
    /// O(1) in the common case: just increments the epoch counter.
    /// O(capacity) only when `currentEpoch` reaches `UInt16.max`, which happens
    /// at most once every 65,534 `nextEpoch()` calls per `VisitedList` instance.
    /// In practice (1–6 `searchLayer` calls per insert/search), this path is never hit.
    mutating func nextEpoch() {
        if currentEpoch == UInt16.max {
            // Reset to avoid wraparound collision: a slot that still holds an old
            // epoch value (e.g. 500) would appear "visited" once the counter cycles
            // back to 500 — a silent recall bug. Full zeroing is the safe reset.
            for i in epochs.indices { epochs[i] = 0 }
            currentEpoch = 1
        } else {
            currentEpoch &+= 1
        }
    }

    /// Returns `true` if `id` was inserted in the current epoch.
    @inline(__always)
    func contains(_ id: Int32) -> Bool {
        let idx = Int(id)
        guard idx >= 0, idx < epochs.count else { return false }
        return epochs[idx] == currentEpoch
    }

    /// Records `id` as visited in the current epoch.
    @inline(__always)
    mutating func insert(_ id: Int32) {
        let idx = Int(id)
        guard idx >= 0, idx < epochs.count else { return }
        epochs[idx] = currentEpoch
    }
}
