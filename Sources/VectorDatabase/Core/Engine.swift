import Foundation

/// Engine.swift — The non-actor core engine.
///
/// WHY ENGINE IS NOT AN ACTOR (§9 Pitfalls):
/// If `Engine` were an actor, its methods would be isolated. We need to acquire
/// a `ReadWriteLock` and then perform synchronous mutations. If an actor suspended
/// while holding a lock (e.g. across `await`), it would violate actor reentrancy rules
/// and risk deadlocks or memory corruption. By making `Engine` a standard class,
/// we can enforce strict, non-yielding synchronous critical sections via `withRead`
/// and `withWrite`, and then wrap the entire `Engine` in the public `VectorDatabase` actor
/// which provides the safe async boundary.
///
/// WHY ENGINE IS SAFE AS UNCHECKED SENDABLE (§9):
/// All internal mutable state accesses (reads and writes) are protected by a single
/// instance of `ReadWriteLock`. This guarantees that even when methods are called
/// concurrently from different Tasks (e.g. background saving vs. foreground inserts),
/// no data races can occur.
final class Engine: @unchecked Sendable {
    private let rwlock = ReadWriteLock()
    private var router: IndexRouter
    private var isRebuilding = false

    init(
        dimension: Int, metric: DistanceMetric, hnswParams: HNSWParameters = .default,
        hnswThreshold: Int = 2000
    ) {
        self.router = IndexRouter(
            dimension: dimension, metric: metric, hnswParams: hnswParams,
            hnswThreshold: hnswThreshold)
    }

    init(hnswIndex: HNSWIndex, hnswThreshold: Int = 2000) {
        self.router = IndexRouter(hnswIndex: hnswIndex, hnswThreshold: hnswThreshold)
    }

    // MARK: - Synchronous API (Locking Boundary)

    func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
        // [Pitfall 3]: ensuring grow() on VectorStorage/GraphStorage only ever happens under the write lock.
        // We acquire the exclusive write lock here, ensuring no readers can observe dangling pointers
        // during or after a reallocation.
        try rwlock.withWrite {
            try router.insert(internalID: internalID, vector: vector)
        }
    }

    func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)]
    {
        // Reads acquire the shared read lock. Multiple readers can execute concurrently.
        return rwlock.withRead {
            return router.search(query: query, k: k, ef: ef)
        }
    }

    func getVector(internalID: Int32) -> [Float]? {
        return rwlock.withRead {
            return router.getVector(internalID: internalID)
        }
    }

    func remove(internalID: Int32) throws {
        try rwlock.withWrite {
            try router.remove(internalID: internalID)
        }
    }

    var dimension: Int {
        return router.dimension
    }

    var metric: DistanceMetric {
        return router.metric
    }

    var tombstonedCount: Int {
        return rwlock.withRead {
            return router.tombstonedCount
        }
    }

    func exportHNSWIndex() -> HNSWIndex {
        return rwlock.withRead {
            return router.exportHNSWIndex()
        }
    }

    func close() {
        rwlock.withWrite {
            router.close()
        }
    }

    var stats: IndexStats {
        return rwlock.withRead {
            return router.stats
        }
    }

    // MARK: - Rebuild (§11)

    var shouldRebuild: Bool {
        return rwlock.withRead {
            let s = router.stats
            guard s.isUsingHNSW, s.liveCount > 0 else { return false }
            // Guide §11: trigger rebuild once tombstoned ratio crosses 10–20% threshold.
            // We use >= 0.20 (inclusive upper bound of the suggested range).
            return Double(s.tombstonedCount) / Double(s.liveCount + s.tombstonedCount) >= 0.20
        }
    }

    /// Rebuilds the index from an OWNED snapshot of live vectors, then swaps the
    /// fresh router in under the write lock.
    ///
    /// CALLER CONTRACT (fix K2): the caller MUST guarantee that no insert, update
    /// or delete can interleave between `collectLiveSnapshots()` and this call's
    /// return. Any mutation applied to the OLD router in that window is silently
    /// discarded by the swap in step 4 — it exists only in the old router, which
    /// is thrown away. `VectorDatabase` satisfies this by calling both from the same,
    /// uninterrupted actor-isolated turn (no `await` in between).
    ///
    /// Elements are (internalID, originalLevel, vectorData).
    func rebuild(with liveVectors: [(id: Int32, level: Int, vector: [Float])]) throws {
        // 1. Guard against concurrent rebuilds and capture configuration safely
        let setupData:
            (
                canRebuild: Bool, dimension: Int, metric: DistanceMetric,
                hnswParams: HNSWParameters, hnswThreshold: Int
            )? = rwlock.withWrite {
                guard !isRebuilding else { return nil }
                isRebuilding = true
                return (
                    true, router.dimension, router.metric, router.hnswParams, router.hnswThreshold
                )
            }

        guard let setup = setupData else { return }

        defer {
            rwlock.withWrite { isRebuilding = false }
        }

        // 2. Build the fresh index outside the main write lock so readers are
        //    not blocked for the whole O(live count) reconstruction.
        var newRouter = IndexRouter(
            dimension: setup.dimension,
            metric: setup.metric,
            hnswParams: setup.hnswParams,
            hnswThreshold: setup.hnswThreshold)

        // 3. Sort by original level descending to preserve HNSW graph quality (§11)
        let sorted = liveVectors.sorted { $0.level > $1.level }

        for item in sorted {
            try item.vector.withUnsafeBufferPointer { buf in
                guard let ptr = buf.baseAddress else { return }
                try newRouter.insert(internalID: item.id, vector: ptr)
            }
        }

        // 4. Atomically swap under the write lock
        rwlock.withWrite {
            self.router = newRouter
        }
    }

    // MARK: - Snapshot extraction

    /// Copies the live vector/graph data directly into a caller-supplied
    /// destination (typically an `mmap`'d file, see
    /// `PersistenceManager.beginSave`) instead of into new heap `Data` buffers.
    ///
    /// `makeDestination` runs SYNCHRONOUSLY inside the same read-lock
    /// acquisition as the copy itself — it is expected to be fast (just `open`/
    /// `ftruncate`/`mmap` syscalls, not proportional to vector count).
    func writeSnapshot(
        makeDestination: (SnapshotSectionSizes) throws -> SnapshotDestination
    ) throws -> MappedSnapshotMetadata {
        try rwlock.withRead {
            try router.writeSnapshot(makeDestination: makeDestination)
        }
    }

    /// Returns an owned snapshot of all live (non-tombstoned) vectors.
    ///
    /// Each element is `(internalID, originalLevel, vector[Float])`.
    /// Vector bytes are copied into Swift-managed `[Float]` arrays, so the snapshot
    /// is NOT invalidated if `VectorStorage` grows (and remaps) concurrently.
    ///
    /// Call this immediately before `rebuild(with:)` and pass the result straight
    /// to it — see the caller contract on `rebuild`.
    func collectLiveSnapshots() -> [(id: Int32, level: Int, vector: [Float])] {
        return rwlock.withRead {
            router.collectLiveSnapshots()
        }
    }

    // MARK: - Inspector Methods

    func inspectEntryPoint() -> Int32? {
        return rwlock.withRead {
            return router.inspectEntryPoint()
        }
    }

    func inspectNodeLevel(internalID: Int32) -> Int? {
        return rwlock.withRead {
            return router.inspectNodeLevel(internalID: internalID)
        }
    }

    func inspectNeighbors(internalID: Int32, atLayer layer: Int) -> [Int32]? {
        return rwlock.withRead {
            return router.inspectNeighbors(internalID: internalID, atLayer: layer)
        }
    }

    func inspectFullTopology() -> (
        entryPoint: Int32?, entryPointLevel: Int,
        nodes: [Int32: (level: Int, neighborsByLayer: [[Int32]])]
    ) {
        return rwlock.withRead {
            return router.inspectFullTopology()
        }
    }
}
