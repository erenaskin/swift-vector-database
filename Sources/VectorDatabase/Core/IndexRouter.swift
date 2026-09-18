/// IndexRouter.swift — Internal index dispatch layer (§8 Pitfall: Small Collections).
///
/// PITFALL: Small-collection overhead (§8 Pitfalls).
///   HNSW has real overhead — graph construction, per-node heap allocations — that is
///   not justified below a few thousand vectors. A user with 300 notes should not pay
///   O(M * efConstruction) insert cost for each new note.
///
/// SOLUTION: `IndexRouter` transparently uses `FlatIndex` below `hnswThreshold` and
///   switches to `HNSWIndex` above it. The public API (`VectorDatabase` actor)
///   delegates to `IndexRouter`, so callers never observe the switch.
///
/// MIGRATION STRATEGY:
///   When `count` crosses `hnswThreshold`, `IndexRouter` migrates the existing
///   FlatIndex vectors into a freshly-built `HNSWIndex`. This is a one-time
///   O(n * efConstruction) rebuild, amortized across subsequent inserts.
///   The switch is permanent for the lifetime of the `Engine`/`IndexRouter` instance.
///
/// NOTE: This type lives in the library-internal layer and is NOT `public`.
///       The `VectorDatabase` actor exposes the public surface.

import Foundation

public struct IndexStats: Sendable {
    public let liveCount: Int
    public let tombstonedCount: Int
    public let isUsingHNSW: Bool
}

struct IndexRouter: Sendable {

    // MARK: - Configuration

    /// Number of vectors below which FlatIndex is preferred over HNSWIndex.
    /// Default: 2,000 per §8 Pitfalls recommendation.
    let hnswThreshold: Int

    let dimension: Int
    let metric: DistanceMetric
    let hnswParams: HNSWParameters

    // MARK: - State

    private enum ActiveIndex: Sendable {
        case flat(FlatIndex)
        case hnsw(HNSWIndex)
    }

    private var index: ActiveIndex

    /// Number of PHYSICAL rows handed to the active index, used only to decide
    /// when to migrate from flat to HNSW. Not a live count — see `stats`.
    private var insertedCount: Int = 0

    // MARK: - Init

    init(
        dimension: Int,
        metric: DistanceMetric,
        hnswParams: HNSWParameters = .default,
        hnswThreshold: Int = 2_000
    ) {
        self.dimension = dimension
        self.metric = metric
        self.hnswParams = hnswParams
        self.hnswThreshold = hnswThreshold
        self.index = .flat(FlatIndex(dimension: dimension, metric: metric))
    }

    init(hnswIndex: HNSWIndex, hnswThreshold: Int = 2_000) {
        self.dimension = hnswIndex.dimension
        self.metric = hnswIndex.metric
        self.hnswParams = hnswIndex.params
        self.hnswThreshold = hnswThreshold
        self.index = .hnsw(hnswIndex)
        // `slotCount`, not `count`: this counter tracks physical rows, and
        // `HNSWIndex.count` is now the LIVE count (tombstones excluded).
        self.insertedCount = hnswIndex.slotCount
    }

    // MARK: - VectorIndex forwarding

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

    func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)]
    {
        switch index {
        case .flat(let flat): return flat.search(query: query, k: k, ef: ef)
        case .hnsw(let hnsw): return hnsw.search(query: query, k: k, ef: ef)
        }
    }

    func getVector(internalID: Int32) -> [Float]? {
        switch index {
        case .flat(let flat): return flat.getVector(internalID: internalID)
        case .hnsw(let hnsw): return hnsw.getVector(internalID: internalID)
        }
    }

    mutating func remove(internalID: Int32) throws {
        switch index {
        case .flat(var flat):
            // Hard removal: the slot is physically reclaimed via swap-remove.
            // Decrement insertedCount to keep the migration counter accurate.
            try flat.remove(internalID: internalID)
            index = .flat(flat)
            insertedCount -= 1
        case .hnsw(var hnsw):
            // Soft-delete (tombstone): the node's slot in VectorStorage / GraphStorage
            // is NOT reclaimed. Other nodes' edges may still traverse through it.
            // Do NOT decrement insertedCount — the physical slot count is unchanged.
            try hnsw.remove(internalID: internalID)
            index = .hnsw(hnsw)
        }
    }

    var tombstonedCount: Int {
        switch index {
        case .flat: return 0  // FlatIndex does not use tombstones
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
    //
    // FIX (coverage pass): this used to also declare an instance-level
    // `var isUsingHNSW: Bool { if case .hnsw = index { true } else { false } }`
    // right here. It had zero call sites anywhere in Sources or Tests —
    // confirmed by grep, not just by the coverage report showing 0 executions
    // — and it silently shadowed the UNRELATED `IndexStats.isUsingHNSW` stored
    // property below (same name, different type, genuinely used by
    // `Engine.shouldRebuild` as `router.stats.isUsingHNSW`). Keeping a dead
    // property with the same name as a live one is a footgun for the next
    // person who reaches for `router.isUsingHNSW` expecting it to be the real
    // one, so it was removed rather than given a test that would only exist
    // to inflate a coverage number. If a genuine need for a router-level
    // (rather than stats-level) mode check shows up later, `stats.isUsingHNSW`
    // already answers it without a second, easily-confused entry point.

    var stats: IndexStats {
        switch index {
        case .flat(let flat):
            return IndexStats(liveCount: flat.count, tombstonedCount: 0, isUsingHNSW: false)
        case .hnsw(let hnsw):
            // `HNSWIndex.count` is itself the live count now, so there is nothing
            // left to subtract here — doing so would double-count tombstones.
            return IndexStats(
                liveCount: hnsw.count,
                tombstonedCount: hnsw.tombstoned.count,
                isUsingHNSW: true)
        }
    }

    /// Returns an owned snapshot of all live (non-tombstoned) nodes for a pre-rebuild copy.
    /// Only meaningful when backed by HNSW; returns [] for FlatIndex (no tombstone concept).
    func collectLiveSnapshots() -> [(id: Int32, level: Int, vector: [Float])] {
        guard case .hnsw(let hnsw) = index else { return [] }
        var result: [(id: Int32, level: Int, vector: [Float])] = []
        // `nodeSlots.count` is O(1) — no gaps exist because HNSW never physically
        // removes a slot, so the slot array length equals the occupied slot count.
        result.reserveCapacity(hnsw.nodeSlots.count - hnsw.tombstoned.count)
        for (idx, slot) in hnsw.nodeSlots.enumerated() {
            guard let node = slot else { continue }
            let id = Int32(idx)
            guard !hnsw.tombstoned.contains(id) else { continue }
            let ptr = hnsw.vectorStorage.pointer(toSlot: Int(node.vectorSlot))
            let vec = Array(UnsafeBufferPointer(start: ptr, count: hnsw.dimension))
            result.append((id: id, level: node.level, vector: vec))
        }
        return result
    }

    // MARK: - Snapshot writing (the one and only persistence path)

    /// Computes section sizes, asks `makeDestination` to size + `mmap` a
    /// destination file and return raw pointers into it, then copies the live
    /// vector/graph data DIRECTLY into those pointers — all synchronously,
    /// within one call (the caller, `Engine`, wraps this in a single read-lock
    /// acquisition). No intermediate heap `Data` copy of the (potentially very
    /// large) vector/graph sections is ever made.
    ///
    /// FIX S1: this used to be the *second* of two snapshot implementations.
    /// `createSnapshot()` built full heap `Data` blobs for
    /// `PersistenceManager.save(snapshot:idMap:)`, which production code stopped
    /// calling once `beginSave`/`finishSave` landed — only tests still reached
    /// it. Two independent implementations of the same on-disk layout (including
    /// two different ways of computing the file checksum) is a correctness trap,
    /// so the heap-copy path and its `IndexSnapshot` value type are gone.
    ///
    /// SIZING NOTE: sections are sized by `count`, NOT `capacity`. Geometric
    /// growth means `capacity` can be up to 2x larger than what is in use, and
    /// every node's `vectorSlot` is a dense index in `0..<count` (HNSW never
    /// physically removes rows — only tombstones them — so there are no gaps).
    /// `PersistenceManager.load()` mirrors this by reconstructing the mapped
    /// storages with `capacity == header.vectorCount`.
    func writeSnapshot(
        makeDestination: (SnapshotSectionSizes) throws -> SnapshotDestination
    ) throws -> MappedSnapshotMetadata {
        let hnsw = exportHNSWIndex()

        let vectorBytes = hnsw.vectorStorage.count * hnsw.dimension * MemoryLayout<Float>.size
        let l0Bytes =
            hnsw.graphStorage.count * hnsw.graphStorage.mMax0 * MemoryLayout<Int32>.size
        let L = hnsw.graphStorage.neighborCounts.count - 1
        var upperByteSizes: [Int] = []
        if L > 0 {
            for _ in 1...L {
                upperByteSizes.append(
                    hnsw.graphStorage.count * hnsw.graphStorage.m * MemoryLayout<Int32>.size)
            }
        }

        let sizes = SnapshotSectionSizes(
            vectorBytes: vectorBytes, layer0Bytes: l0Bytes, upperLayerByteSizes: upperByteSizes)
        let destination = try makeDestination(sizes)

        precondition(
            destination.upperLayers.count == upperByteSizes.count,
            "makeDestination must return exactly one pointer per upper layer")

        if vectorBytes > 0 {
            destination.vector.copyMemory(from: hnsw.vectorStorage.buffer, byteCount: vectorBytes)
        }
        if l0Bytes > 0 {
            destination.layer0.copyMemory(from: hnsw.graphStorage.layer0, byteCount: l0Bytes)
        }
        for i in 0..<upperByteSizes.count where upperByteSizes[i] > 0 {
            destination.upperLayers[i].copyMemory(
                from: hnsw.graphStorage.upperLayerPointer(i + 1), byteCount: upperByteSizes[i])
        }

        return MappedSnapshotMetadata(
            nodes: hnsw.nodesDictionary,
            entryPoint: hnsw.entryPoint,
            entryPointLevel: hnsw.entryPointLevel,
            neighborCounts: hnsw.graphStorage.neighborCounts,
            hnswSeed: hnsw.params.seed,
            hnswEfConstruction: hnsw.params.efConstruction,
            hnswEfSearch: hnsw.params.efSearch,
            walFormatVersion: 2,
            metric: hnsw.metric,
            vectorCount: hnsw.vectorStorage.count,
            capacity: hnsw.graphStorage.capacity,
            mMax0: hnsw.graphStorage.mMax0,
            m: hnsw.graphStorage.m,
            L: L
        )
    }

    // MARK: - Inspector Methods

    func inspectEntryPoint() -> Int32? {
        guard case .hnsw(let hnsw) = index else { return nil }
        // The entryPoint in HNSWIndex could be nil if the graph is empty or entirely tombstoned
        return hnsw.entryPoint
    }

    func inspectNodeLevel(internalID: Int32) -> Int? {
        guard case .hnsw(let hnsw) = index else { return nil }
        guard let node = hnsw.node(for: internalID), !hnsw.tombstoned.contains(internalID) else {
            return nil
        }
        return node.level
    }

    func inspectNeighbors(internalID: Int32, atLayer layer: Int) -> [Int32]? {
        guard case .hnsw(let hnsw) = index else { return nil }
        guard let node = hnsw.node(for: internalID), !hnsw.tombstoned.contains(internalID) else {
            return nil
        }
        guard layer >= 0, layer <= node.level else { return nil }

        let rawNeighbors = hnsw.graphStorage.neighbors(of: node.vectorSlot, at: layer)
        return rawNeighbors.filter { !hnsw.tombstoned.contains($0) }
    }

    func inspectFullTopology() -> (
        entryPoint: Int32?, entryPointLevel: Int,
        nodes: [Int32: (level: Int, neighborsByLayer: [[Int32]])]
    ) {
        guard case .hnsw(let hnsw) = index else { return (nil, 0, [:]) }

        var nodeDict: [Int32: (level: Int, neighborsByLayer: [[Int32]])] = [:]
        for (idx, slot) in hnsw.nodeSlots.enumerated() {
            guard let node = slot else { continue }
            let id = Int32(idx)
            guard !hnsw.tombstoned.contains(id) else { continue }
            var allNeighbors: [[Int32]] = []
            for layer in 0...node.level {
                let raw = hnsw.graphStorage.neighbors(of: node.vectorSlot, at: layer)
                let live = raw.filter { !hnsw.tombstoned.contains($0) }
                allNeighbors.append(live)
            }
            nodeDict[id] = (level: node.level, neighborsByLayer: allNeighbors)
        }

        return (entryPoint: hnsw.entryPoint, entryPointLevel: hnsw.entryPointLevel, nodes: nodeDict)
    }
}
