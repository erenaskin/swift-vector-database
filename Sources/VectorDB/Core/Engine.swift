import Foundation

/// Engine.swift — Phase 5: The non-actor core engine.
///
/// WHY ENGINE IS NOT AN ACTOR (§9 Pitfalls):
/// If `Engine` were an actor, its methods would be isolated. We need to acquire 
/// a `ReadWriteLock` and then perform synchronous mutations. If an actor suspended
/// while holding a lock (e.g. across `await`), it would violate actor reentrancy rules 
/// and risk deadlocks or memory corruption. By making `Engine` a standard class, 
/// we can enforce strict, non-yielding synchronous critical sections via `withRead` 
/// and `withWrite`, and then wrap the entire `Engine` in the public `VectorDB` actor 
/// which provides the safe async boundary.
final class Engine {
    private let rwlock = ReadWriteLock()
    private var router: IndexRouter
    private var isRebuilding = false

    init(dimension: Int, metric: DistanceMetric, hnswParams: HNSWParameters = .default, hnswThreshold: Int = 2000) {
        self.router = IndexRouter(dimension: dimension, metric: metric, hnswParams: hnswParams, hnswThreshold: hnswThreshold)
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

    func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)] {
        // Reads acquire the shared read lock. Multiple readers can execute concurrently.
        return rwlock.withRead {
            return router.search(query: query, k: k, ef: ef)
        }
    }

    func remove(internalID: Int32) throws {
        try rwlock.withWrite {
            try router.remove(internalID: internalID)
        }
    }
    
    var count: Int {
        return rwlock.withRead {
            return router.count
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

    // MARK: - Updates & Rebuilds (Phase 7)

    /// Implements `update` as tombstone-old + insert-fresh, exactly as the guide specifies.
    func update(internalID: Int32, vector: UnsafePointer<Float>) throws {
        try rwlock.withWrite {
            // We ignore remove errors (e.g. notFound) because we want to ensure
            // the new vector is inserted regardless.
            try? router.remove(internalID: internalID)
            try router.insert(internalID: internalID, vector: vector)
        }
    }

    var shouldRebuild: Bool {
        return rwlock.withRead {
            let s = router.stats
            guard s.isUsingHNSW, s.liveCount > 0 else { return false }
            // Guide §11: trigger rebuild once tombstoned ratio crosses 10–20% threshold.
            // We use >= 0.20 (inclusive upper bound of the suggested range).
            return Double(s.tombstonedCount) / Double(s.liveCount + s.tombstonedCount) >= 0.20
        }
    }

    /// Safely performs a background rebuild. 
    /// Takes an OWNED snapshot of live vectors to avoid dangling pointers if `VectorStorage` grows concurrently.
    /// Elements are (internalID, originalLevel, vectorData).
    func rebuild(with liveVectors: [(id: Int32, level: Int, vector: [Float])]) throws {
        // 1. Guard against concurrent rebuilds
        let canRebuild = rwlock.withWrite { () -> Bool in
            guard !isRebuilding else { return false }
            isRebuilding = true
            return true
        }
        guard canRebuild else { return }

        defer {
            rwlock.withWrite { isRebuilding = false }
        }

        // 2. Build fresh index off the main actor / outside the main write lock
        var newRouter = IndexRouter(dimension: router.dimension, 
                                    metric: router.metric, 
                                    hnswParams: router.hnswParams, 
                                    hnswThreshold: router.hnswThreshold)

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

    // MARK: - Snapshot extraction (for rebuild, testing)

    /// Returns an owned snapshot of all live (non-tombstoned) vectors.
    ///
    /// Each element is `(internalID, originalLevel, vector[Float])`.
    /// Vector bytes are copied into Swift-managed `[Float]` arrays, so the snapshot
    /// is NOT invalidated if `VectorStorage` grows (and remaps) concurrently.
    ///
    /// Delegates to `IndexRouter.collectLiveSnapshots()`, keeping IndexRouter's
    /// private state (index enum, HNSWIndex internals) fully encapsulated.
    /// Call this *before* `rebuild(with:)` and pass the result straight to it.
    func collectLiveSnapshots() -> [(id: Int32, level: Int, vector: [Float])] {
        return rwlock.withRead {
            router.collectLiveSnapshots()
        }
    }
}
