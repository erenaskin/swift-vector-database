/// IndexRouter.swift — Internal index dispatch layer (§8 Pitfall: Small Collections).
///
/// PITFALL: Small-collection overhead (§8 Pitfalls).
///   HNSW has real overhead — graph construction, per-node heap allocations — that is
///   not justified below a few thousand vectors. A user with 300 notes should not pay
///   O(M * efConstruction) insert cost for each new note.
///
/// SOLUTION: `IndexRouter` transparently uses `FlatIndex` below `hnswThreshold` and
///   switches to `HNSWIndex` above it. The public API (`VectorDB` actor, Phase 5+8)
///   delegates to `IndexRouter`, so callers never observe the switch.
///
/// MIGRATION STRATEGY:
///   When `count` crosses `hnswThreshold` from above (on the next insert), `IndexRouter`
///   migrates the existing FlatIndex vectors into a freshly-built `HNSWIndex`. This is
///   a one-time O(n * efConstruction) rebuild, amortized across subsequent inserts.
///   The switch is permanent for the lifetime of the `Engine`/`IndexRouter` instance;
///   there is no downward migration (vectors are never deleted en-masse in Phase 4).
///
/// DELETION PITFALL NOTE (§8 Pitfalls — Phase 7):
///   If the entry point node is deleted, `HNSWIndex.search` will silently produce wrong
///   results (or crash) because it traverses from `entryPoint` without validating that
///   the node still exists. Phase 7 will address this with a soft-delete + rebuild
///   strategy. `IndexRouter` reserves space in its design:
///     - All node IDs go through `IndexRouter.idToSlot` (in `IDMap`, Phase 7), giving
///       Phase 7 a natural hook to intercept deletions and trigger re-entry-point selection.
///     - `HNSWIndex.remove` already throws `.notFound` rather than silently corrupting
///       state, making the transition to HNSW unambiguous.
///
/// NOTE: This type lives in the library-internal layer and is NOT `public`.
///       The `VectorDB` actor (Phase 5) exposes the public surface.

public struct IndexStats {
    public let liveCount: Int
    public let tombstonedCount: Int
    public let isUsingHNSW: Bool
}

struct IndexRouter {

    // MARK: - Configuration

    /// Number of vectors below which FlatIndex is preferred over HNSWIndex.
    /// Default: 2,000 per §8 Pitfalls recommendation.
    let hnswThreshold: Int

    let dimension: Int
    let metric: DistanceMetric
    let hnswParams: HNSWParameters

    // MARK: - State

    private enum ActiveIndex {
        case flat(FlatIndex)
        case hnsw(HNSWIndex)
    }

    private var index: ActiveIndex
    private var insertedCount: Int = 0

    // MARK: - Init

    init(dimension: Int,
         metric: DistanceMetric,
         hnswParams: HNSWParameters = .default,
         hnswThreshold: Int = 2_000) {
        self.dimension      = dimension
        self.metric         = metric
        self.hnswParams     = hnswParams
        self.hnswThreshold  = hnswThreshold
        self.index = .flat(FlatIndex(dimension: dimension, metric: metric))
    }
    
    init(hnswIndex: HNSWIndex, hnswThreshold: Int = 2_000) {
        self.dimension = hnswIndex.dimension
        self.metric = hnswIndex.metric
        self.hnswParams = hnswIndex.params
        self.hnswThreshold = hnswThreshold
        self.index = .hnsw(hnswIndex)
        self.insertedCount = hnswIndex.count
    }

    // MARK: - VectorIndex forwarding

    var count: Int { insertedCount }

    mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
        // Migrate from FlatIndex to HNSWIndex the first time we cross the threshold.
        if case .flat(var flat) = index, insertedCount == hnswThreshold {
            var hnsw = HNSWIndex(dimension: dimension, metric: metric, params: hnswParams)
            // Delegate re-insertion to FlatIndex so private storage/idToSlot stay encapsulated.
            flat.reinsertInto(hnsw: &hnsw)
            index = .hnsw(hnsw)
        }

        switch index {
        case .flat(var flat):
            try flat.insert(internalID: internalID, vector: vector)
            index = .flat(flat)
        case .hnsw(var hnsw):
            try hnsw.insert(internalID: internalID, vector: vector)
            index = .hnsw(hnsw)
        }
        insertedCount += 1
    }

    func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)] {
        switch index {
        case .flat(let flat): return flat.search(query: query, k: k, ef: ef)
        case .hnsw(let hnsw): return hnsw.search(query: query, k: k, ef: ef)
        }
    }

    mutating func remove(internalID: Int32) throws {
        switch index {
        case .flat(var flat):
            // Hard removal: the slot is physically reclaimed via swap-remove.
            // Decrement insertedCount to keep `count` accurate.
            try flat.remove(internalID: internalID)
            index = .flat(flat)
            insertedCount -= 1
        case .hnsw(var hnsw):
            // Soft-delete (tombstone): the node's slot in VectorStorage / GraphStorage
            // is NOT reclaimed. Other nodes' edges may still traverse through it.
            // Do NOT decrement insertedCount — the physical slot count is unchanged.
            // liveCount is tracked separately via hnsw.tombstoned.count in `stats`.
            try hnsw.remove(internalID: internalID)
            index = .hnsw(hnsw)
        }
    }
    
    var tombstonedCount: Int {
        switch index {
        case .flat: return 0 // FlatIndex does not use tombstones
        case .hnsw(let hnsw): return hnsw.tombstoned.count
        }
    }
    
    func exportHNSWIndex() -> HNSWIndex {
        switch index {
        case .hnsw(let hnsw):
            return hnsw
        case .flat(var flat):
            // Fallback: migrate everything into a new HNSW index and return it.
            var hnsw = HNSWIndex(dimension: dimension, metric: metric, params: hnswParams)
            flat.reinsertInto(hnsw: &hnsw)
            return hnsw
        }
    }
    
    mutating func close() {
        self.index = .flat(FlatIndex(dimension: dimension, metric: metric))
    }

    // MARK: - Inspection

    var isUsingHNSW: Bool {
        if case .hnsw = index { return true }
        return false
    }

    var stats: IndexStats {
        switch index {
        case .flat(let flat):
            return IndexStats(liveCount: flat.count, tombstonedCount: 0, isUsingHNSW: false)
        case .hnsw(let hnsw):
            return IndexStats(liveCount: hnsw.count - hnsw.tombstoned.count,
                              tombstonedCount: hnsw.tombstoned.count,
                              isUsingHNSW: true)
        }
    }

    /// Returns an owned snapshot of all live (non-tombstoned) nodes for a pre-rebuild copy.
    /// Only meaningful when backed by HNSW; returns [] for FlatIndex (no tombstone concept).
    func collectLiveSnapshots() -> [(id: Int32, level: Int, vector: [Float])] {
        guard case .hnsw(let hnsw) = index else { return [] }
        var result: [(id: Int32, level: Int, vector: [Float])] = []
        result.reserveCapacity(hnsw.nodes.count - hnsw.tombstoned.count)
        for (id, node) in hnsw.nodes {
            guard !hnsw.tombstoned.contains(id) else { continue }
            let ptr = hnsw.vectorStorage.pointer(toSlot: Int(node.vectorSlot))
            let vec = Array(UnsafeBufferPointer(start: ptr, count: hnsw.dimension))
            result.append((id: id, level: node.level, vector: vec))
        }
        return result
    }
}
