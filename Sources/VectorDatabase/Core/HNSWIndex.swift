/// HNSWIndex.swift — Approximate nearest-neighbor graph engine (§8).
///
/// Implements §8.4 (randomLevel), §8.5 (searchLayer), §8.6 (insert +
/// selectNeighborsHeuristic), and §8.7 (search/query).
///
/// ARCHITECTURE:
///   - `vectorStorage: VectorStorage` — raw float data
///   - `graphStorage: GraphStorage`   — fixed-width adjacency lists
///   - `nodeSlots: [HNSWNode?]`       — dense array mapping internalID → (level, vectorSlot)
///   - `rng: SeedableRNG`             — seeded PRNG for reproducible level assignment
///
/// NODE STORAGE (PERFORMANCE NOTE):
///   `nodeSlots` replaced the earlier `[Int32: HNSWNode]` dictionary to eliminate
///   hash-based lookup overhead in the hot path (`searchLayer`, `scored`,
///   `selectNeighborsHeuristic`). Since `internalID` is a monotonically increasing
///   `Int32`, a direct-indexed array provides O(1) access with no hashing or
///   collision resolution. Trade-off: deleted nodes leave `nil` slots that are
///   never reclaimed until a rebuild — see `nodesDictionary` doc comment.
///
/// SEEDED RNG (§8.4 / §8 Pitfalls):
///   "does my recall regression test still pass" is meaningless if the graph
///   structure changes on every run. HNSWIndex always uses `SeedableRNG` — the
///   same xorshift64 used in TestFixtures — seeded from `params.seed`. The seed
///   is part of the serialised format so the graph can be reproduced
///   identically after a reload.
///
/// HEAP CONVENTION (§8.5 Pitfall):
///   This codebase uses "higher score = more similar" throughout. The HNSW paper
///   uses "smaller distance = closer". The two heaps in `searchLayer` are:
///     · candidateHeap: pop HIGHEST score first  → explore closest candidates first
///     · foundHeap:     pop LOWEST  score first  → evict FARTHEST when set is full

import Foundation

/// Note: This type is marked `@unchecked Sendable` to allow advanced users to bypass the `VectorDatabase` actor
/// for direct parallel reads. It is NOT internally thread-safe. Callers must synchronize all access.
public struct HNSWIndex: VectorIndex, @unchecked Sendable {

    // MARK: - Stored properties

    public let dimension: Int
    public let metric: DistanceMetric
    public let params: HNSWParameters

    internal var vectorStorage: VectorStorage
    internal var graphStorage: GraphStorage

    /// Dense array mapping internalID → node metadata.
    /// `nodeSlots[Int(internalID)]` is non-nil for every live or tombstoned node,
    /// and nil for IDs that were never inserted or for gaps.
    ///
    /// TRADE-OFF: internalIDs are never reused (nextID is monotonically
    /// increasing), so `nodeSlots.count` grows to `maxHistoricalID + 1`
    /// regardless of how many nodes have been deleted. In high-churn scenarios
    /// (many insert+delete cycles without rebuild), this uses more memory than
    /// a dictionary. This is an accepted trade-off; rebuild-time ID renumbering
    /// would be a much larger, riskier change (touching IDMap, WAL replay, and
    /// persistence) and is out of scope.
    private(set) var nodeSlots: [HNSWNode?] = []

    /// Tracks soft-deleted internal IDs (§11).
    var tombstoned: Set<Int32> = []

    /// Current global entry point (the node at the topmost layer).
    var entryPoint: Int32? = nil
    var entryPointLevel: Int = 0

    /// Seeded RNG for deterministic level assignment (§8.4).
    /// MUST NOT be replaced with the global `Double.random` — see file header.
    public var rng: SeedableRNG

    // MARK: - Node accessors

    /// O(1) lookup of a node by its internalID. Returns nil if the ID is out of
    /// range or was never inserted.
    @inline(__always)
    func node(for internalID: Int32) -> HNSWNode? {
        let idx = Int(internalID)
        guard idx >= 0, idx < nodeSlots.count else { return nil }
        return nodeSlots[idx]
    }

    /// Stores a node at the given internalID, growing the array if needed.
    mutating func setNode(_ node: HNSWNode, for internalID: Int32) {
        let idx = Int(internalID)
        if idx >= nodeSlots.count {
            nodeSlots.append(contentsOf: repeatElement(nil, count: idx - nodeSlots.count + 1))
        }
        nodeSlots[idx] = node
    }

    // MARK: - Serialization bridge

    /// Produces a `[Int32: HNSWNode]` dictionary for serialization (save path).
    /// Called once per save — not performance-critical.
    var nodesDictionary: [Int32: HNSWNode] {
        var dict: [Int32: HNSWNode] = [:]
        dict.reserveCapacity(nodeSlots.count)
        for (idx, slot) in nodeSlots.enumerated() {
            if let n = slot {
                dict[Int32(idx)] = n
            }
        }
        return dict
    }

    /// Bulk-loads nodes from a dictionary (load path). Called once per load.
    mutating func loadNodes(from dict: [Int32: HNSWNode]) {
        nodeSlots = []
        guard let maxID = dict.keys.max() else { return }
        nodeSlots = Array(repeating: nil, count: Int(maxID) + 1)
        for (id, n) in dict {
            nodeSlots[Int(id)] = n
        }
    }

    // MARK: - Counts

    /// The number of LIVE (non-tombstoned) vectors, as `VectorIndex` requires.
    ///
    /// FIX — PROTOCOL CONTRACT VIOLATION:
    /// `VectorIndex.count` is documented as "the number of vectors currently in
    /// the index (live, not tombstoned)". This property used to return
    /// `vectorStorage.count`, i.e. the number of PHYSICAL slots, which never
    /// shrinks because `remove()` is a soft delete. That made `HNSWIndex` and
    /// `FlatIndex` disagree about `count` after a removal — exactly what
    /// `VectorIndexContractTests.testFlatIndexAndHNSWIndexHonorTheSameVectorIndexContract`
    /// asserts must not happen (it failed with "4 is not equal to 3").
    ///
    /// Callers that genuinely need the physical slot count — persistence sizing,
    /// dense slot iteration — must use `slotCount` below.
    public var count: Int { vectorStorage.count - tombstoned.count }

    /// The number of physical rows in `vectorStorage` / `graphStorage`,
    /// including tombstoned ones. HNSW never physically removes a row, so every
    /// node's `vectorSlot` is a dense index in `0..<slotCount`.
    internal var slotCount: Int { vectorStorage.count }

    // MARK: - Init

    public init(dimension: Int, metric: DistanceMetric, params: HNSWParameters = .default) {
        self.dimension = dimension
        self.metric = metric
        self.params = params
        self.vectorStorage = VectorStorage(dimension: dimension)
        self.graphStorage = GraphStorage(m: params.M, mMax0: params.Mmax0)
        self.rng = SeedableRNG(seed: params.seed)
    }

    // MARK: - VectorIndex: insert (§8.6)
    //
    // INSERT PITFALLS ADDRESSED (§8 Pitfalls):
    //
    // [Pitfall-1] Entry-point level must stay correct.
    //   The entry point is updated to the new node ONLY AFTER Phase B completes.
    //   This means every in-progress insert observes the old entry point, which is
    //   stable because inserts are serialised under a write lock. Updating before
    //   Phase B would allow a concurrent reader to traverse from an incompletely-
    //   wired node.
    //
    // [Pitfall-2] Recursive lock reentrancy during neighbour pruning.
    //   `selectNeighborsHeuristic` is called from inside Phase B to prune OTHER
    //   nodes' neighbour lists (bidirectional wiring). A single top-level `insert`
    //   call therefore mutates the graph structure of multiple nodes.
    //   • `Engine` wraps the entire `insert` call in a single write lock acquisition.
    //   • `graphStorage.setNeighbors` does NOT attempt to reacquire the lock — it is
    //     purely an internal struct mutation.
    //   • A non-reentrant pthread_rwlock would deadlock if the pruning step tried to
    //     acquire it again. Our design avoids this by keeping the entire insert —
    //     both primary wiring and pruning — inside one critical section.
    // We assume the caller (Engine) has acquired an exclusive write lock.

    public mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
        let slot = vectorStorage.append(vector)
        let level = randomLevel()

        guard let entryID = entryPoint else {
            // First node: becomes entry point at its own level, no edges.
            entryPoint = internalID
            entryPointLevel = level
            setNode(HNSWNode(level: level, vectorSlot: Int32(slot)), for: internalID)
            graphStorage.addNode()
            return
        }

        setNode(HNSWNode(level: level, vectorSlot: Int32(slot)), for: internalID)
        graphStorage.addNode()

        // VisitedList: allocated ONCE per insert, AFTER nodeSlots has grown to include
        // the new node. Passed inout to searchLayer so each layer calls nextEpoch() for
        // an O(1) reset rather than allocating a new Set<Int32> per layer.
        //
        // DO NOT lift this to an instance variable — doing so would break lock-free
        // concurrent reads. See VisitedList.swift for the full thread-safety rationale.
        var visited = VisitedList(capacity: nodeSlots.count)

        var currentNearest = [entryID]

        // Phase A: descend from top layer to just above the new node's level,
        // greedy ef=1 search at each layer to find a good entry point.
        if level < entryPointLevel {
            for lc in stride(from: entryPointLevel, to: level, by: -1) {
                currentNearest = searchLayer(
                    query: vector,
                    entryPoints: currentNearest,
                    ef: 1,
                    layer: lc,
                    visited: &visited
                ).map(\.id)
            }
        }

        // Phase B: from min(entryPointLevel, level) down to 0, full efConstruction
        // search, neighbor selection, bidirectional wiring, and pruning.
        for lc in stride(from: min(entryPointLevel, level), through: 0, by: -1) {
            let candidates = searchLayer(
                query: vector,
                entryPoints: currentNearest,
                ef: params.efConstruction,
                layer: lc,
                visited: &visited)

            let maxConn = (lc == 0) ? params.Mmax0 : params.M
            let selected = selectNeighborsHeuristic(candidates: candidates, m: maxConn)

            graphStorage.setNeighbors(of: Int32(slot), at: lc, to: selected.map(\.id))

            for neighbor in selected {
                let neighborMax = (lc == 0) ? params.Mmax0 : params.M
                guard let neighborNode = node(for: neighbor.id) else { continue }
                let currentNeighborCount = graphStorage.neighborCount(
                    of: neighborNode.vectorSlot, at: lc)

                if currentNeighborCount < neighborMax {
                    graphStorage.addNeighbor(
                        of: neighborNode.vectorSlot, at: lc, neighborID: internalID)
                } else {
                    // Node is full. Gather existing neighbors + the new one, prune, and overwrite.
                    let neighborVec = vectorStorage.pointer(toSlot: Int(neighborNode.vectorSlot))

                    // P0: Use withNeighbors to avoid heap allocation
                    let pool: [Candidate] = graphStorage.withNeighbors(
                        of: neighborNode.vectorSlot, at: lc
                    ) { neighbors in
                        var p = [Candidate]()
                        p.reserveCapacity(neighbors.count + 1)
                        for i in 0..<neighbors.count {
                            p.append(scored(neighbors[i], neighborVec))
                        }
                        p.append(scored(internalID, neighborVec))
                        return p
                    }

                    let pruned = selectNeighborsHeuristic(candidates: pool, m: neighborMax)
                    graphStorage.setNeighbors(
                        of: neighborNode.vectorSlot, at: lc, to: pruned.map(\.id))
                }
            }

            currentNearest = candidates.map(\.id)
        }

        if level > entryPointLevel {
            entryPoint = internalID
            entryPointLevel = level
        }
    }

    // MARK: - VectorIndex: remove

    public mutating func remove(internalID: Int32) throws {
        guard node(for: internalID) != nil, !tombstoned.contains(internalID) else {
            throw VectorDatabaseError.notFound("internalID \(internalID) not found in HNSWIndex")
        }

        tombstoned.insert(internalID)

        // Pitfall 11.2: correct entry-point reassignment when the current entry point gets tombstoned.
        if internalID == entryPoint {
            // INVARIANT: If `entryPoint` has any living neighbor at `entryPointLevel`, that neighbor MUST
            // have `level >= entryPointLevel` (otherwise the edge couldn't exist). Thus, any such living
            // neighbor can safely become the new `entryPoint` without changing `entryPointLevel`.
            let topNeighbors = neighbors(of: internalID, at: entryPointLevel)
            if let livingNeighbor = topNeighbors.first(where: { !tombstoned.contains($0) }) {
                entryPoint = livingNeighbor
                // entryPointLevel remains unchanged
            } else {
                // Fallback: O(N) scan to find the highest surviving non-tombstoned node.
                var newEntryPoint: Int32? = nil
                var maxLevel = -1

                for (idx, slot) in nodeSlots.enumerated() {
                    guard let n = slot else { continue }
                    let id = Int32(idx)
                    if !tombstoned.contains(id), n.level > maxLevel {
                        maxLevel = n.level
                        newEntryPoint = id
                    }
                }

                if let newEP = newEntryPoint {
                    entryPoint = newEP
                    entryPointLevel = maxLevel
                } else {
                    // The graph is entirely empty of live nodes.
                    entryPoint = nil
                    entryPointLevel = 0
                }
            }
        }
    }

    // MARK: - VectorIndex: search (§8.7)

    public func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(
        id: Int32, score: Float
    )] {
        guard let entryID = entryPoint, k > 0 else { return [] }

        let efSearch = effectiveEf(k: k, requested: ef)

        // VisitedList: one allocation per search, shared across all layers via nextEpoch().
        // search() is a non-mutating func, so this local var does not touch HNSWIndex state.
        var visited = VisitedList(capacity: nodeSlots.count)

        var currentNearest = [entryID]
        for lc in stride(from: entryPointLevel, to: 0, by: -1) {
            currentNearest = searchLayer(
                query: query,
                entryPoints: currentNearest,
                ef: 1,
                layer: lc,
                visited: &visited
            ).map(\.id)
        }
        let results = searchLayer(
            query: query,
            entryPoints: currentNearest,
            ef: efSearch,
            layer: 0,
            visited: &visited)

        // §11: filter out tombstoned nodes, then take k
        let liveResults = results.filter { !tombstoned.contains($0.id) }
        return Array(liveResults.prefix(k)).map { ($0.id, $0.score) }
    }

    /// Chooses the layer-0 candidate list size for a query.
    ///
    /// FIX O1 — WHY THE TOMBSTONE HEADROOM IS NOW BOUNDED:
    /// Tombstoned nodes are still returned by `searchLayer` and filtered out
    /// afterwards, so the candidate list has to over-fetch to still produce `k`
    /// live hits. The old formula was `max(ef, k + tombstoned.count)` — a
    /// completely unbounded term. Because the automatic rebuild only triggers at
    /// a 20% tombstone ratio, a one-million-vector index can legitimately hold
    /// ~250,000 tombstones, which meant `ef` silently became ~250,010: two heaps
    /// of a quarter-million elements, plus an O(N log N) drain, PER QUERY. At
    /// that point HNSW is slower than the brute-force index it exists to replace.
    ///
    /// The headroom is now capped at 3x the base `ef`, so the worst case is a 4x
    /// candidate list rather than an unbounded one. In practice the over-fetch
    /// only has to beat the *local* tombstone density around the query, not the
    /// global tombstone count, and the rebuild threshold keeps that density low.
    private func effectiveEf(k: Int, requested: Int?) -> Int {
        let base = max(requested ?? params.efSearch, k)
        let headroom = min(tombstoned.count, base * 3)
        return base + headroom
    }

    public func getVector(internalID: Int32) -> [Float]? {
        guard let n = node(for: internalID), !tombstoned.contains(internalID) else { return nil }
        let ptr = vectorStorage.pointer(toSlot: Int(n.vectorSlot))
        return Array(UnsafeBufferPointer(start: ptr, count: dimension))
    }

    // MARK: - §8.4 Level Assignment

    /// Malkov & Yashunin's exponential decay level assignment (§8.4).
    ///
    /// Uses the instance-level `rng` (SeedableRNG / xorshift64), NOT the global
    /// `Double.random`. This is mandatory: deterministic graph construction is
    /// required for reproducible recall regression tests.
    mutating func randomLevel() -> Int {
        let r = Double.random(in: 0..<1, using: &rng)
        // Guard: r=0 gives log(0)=-Inf, mL=Inf (when M=1) gives NaN → clamp to 0.
        let raw = -log(max(r, Double.leastNormalMagnitude)) * params.mL
        guard raw.isFinite else { return 0 }
        return Int(floor(raw))
    }

    // MARK: - §8.5 Search Layer (core primitive — used by both insert and query)

    /// Greedy graph traversal within a single layer. Returns up to `ef` candidates
    /// sorted descending by score (highest = most similar, first).
    ///
    /// Heap convention (§8.5 Pitfall — read this carefully):
    ///   · `candidateHeap`: max-heap by score → pop HIGHEST score first
    ///     → we always explore the most promising (closest) unexplored node next.
    ///   · `foundHeap`:     min-heap by score → pop LOWEST  score first
    ///     → we can cheaply check/evict the WORST member of the result set.
    ///
    /// PERFORMANCE FIX (P0): Uses `graphStorage.withNeighbors` for zero-allocation
    /// neighbor traversal instead of `graphStorage.neighbors` which created a
    /// temporary [Int32] array per node visit.
    func searchLayer(
        query: UnsafePointer<Float>,
        entryPoints: [Int32],
        ef: Int,
        layer: Int,
        visited: inout VisitedList
    ) -> [Candidate] {
        guard !entryPoints.isEmpty else { return [] }

        // O(1) epoch reset — no allocation, no memset, just a counter increment.
        visited.nextEpoch()

        // Seed the visited tracker and both heaps from the entry points.
        for ep in entryPoints { visited.insert(ep) }
        let seedCandidates = entryPoints.map { scored($0, query) }
        // candidateHeap: pop CLOSEST (highest score) next — "min-heap by distance" in paper.
        var candidateHeap = BinaryHeap<Candidate>.maxByScore(seedCandidates)
        // foundHeap: peek/pop FARTHEST (lowest score) — "max-heap by distance" in paper.
        var foundHeap = BinaryHeap<Candidate>.minByScore(seedCandidates)

        while let c = candidateHeap.pop() {
            guard let worst = foundHeap.peek() else { break }
            // Early exit: the closest unexplored candidate is farther than our
            // worst result and the result set is already full.
            if c.score < worst.score && foundHeap.count >= ef { break }

            guard let cNode = node(for: c.id) else { continue }

            // P0: Zero-allocation neighbor traversal
            graphStorage.withNeighbors(of: cNode.vectorSlot, at: layer) { neighbors in
                for i in 0..<neighbors.count {
                    let neighborID = neighbors[i]
                    guard !visited.contains(neighborID) else { continue }
                    visited.insert(neighborID)

                    let neighborScore = scored(neighborID, query)
                    if foundHeap.count < ef || neighborScore.score > foundHeap.peek()!.score {
                        candidateHeap.push(neighborScore)
                        foundHeap.push(neighborScore)
                        if foundHeap.count > ef { foundHeap.pop() }
                    }
                }
            }
        }

        // P4: drain and reverse in-place (avoids extra array allocation from .reversed())
        return foundHeap.drainedInPriorityOrderReversed()
    }

    // MARK: - §8.6 Neighbor Selection Heuristic

    /// Diversity-favoring neighbor selection (§8.6). Prefers candidates that are
    /// closer to the query than to any already-selected neighbor. This produces
    /// better-connected graphs than naive top-m selection.
    ///
    /// Backfill: if diversity pruning leaves fewer than `m` selected, fills with
    /// the next-closest remaining candidates.
    ///
    /// PERFORMANCE NOTE (P1 revert — see HeuristicPerformanceTests.swift):
    ///   The previous P1 optimisation used `cblas_sgemv` (batchDot) for diversity checks.
    ///   Micro-benchmarks at m=16/32/64 showed scalar `vDSP_dotpr` is 1.4–1.7x faster
    ///   for all batch sizes typical of this function. `cblas_sgemv` dispatch overhead
    ///   dominates when batch count < ~100. No threshold is needed; full scalar path.
    ///
    ///   M values beyond 128 may eventually tip the balance back to sgemv, but the
    ///   library's target domain (on-device, up to hundreds of thousands of vectors)
    ///   makes such configuration unlikely. This is a known, accepted trade-off.
    private func selectNeighborsHeuristic(candidates: [Candidate], m: Int) -> [Candidate] {
        let sorted = candidates.sorted { $0.score > $1.score }  // closest first
        var selected: [Candidate] = []
        selected.reserveCapacity(m)
        var selectedIDs = Set<Int32>()

        var startIndex = 0
        while startIndex < sorted.count, selected.count < m {
            let candidate = sorted[startIndex]
            startIndex += 1

            guard let candidateNode = node(for: candidate.id) else { continue }
            let candidateVec = vectorStorage.pointer(toSlot: Int(candidateNode.vectorSlot))

            let isDiverse: Bool
            if selected.isEmpty {
                isDiverse = true
            } else {
                // Scalar: one vDSP_dotpr / vDSP_distancesq per already-selected neighbor.
                // Benchmarks show this is 1.4–1.7x faster than cblas_sgemv for m <= 64.
                isDiverse = selected.allSatisfy { existing in
                    guard let existingNode = node(for: existing.id) else { return true }
                    let existingVec = vectorStorage.pointer(
                        toSlot: Int(existingNode.vectorSlot))
                    let distToExisting = VectorMath.similarity(
                        candidateVec, existingVec, dimension, metric: metric)
                    return candidate.score > distToExisting
                }
            }

            if isDiverse {
                selected.append(candidate)
                selectedIDs.insert(candidate.id)
            }
        }

        // Backfill if diversity pruning left us short.
        if selected.count < m {
            for candidate in sorted {
                if selected.count >= m { break }
                if !selectedIDs.contains(candidate.id) {
                    selected.append(candidate)
                    selectedIDs.insert(candidate.id)
                }
            }
        }
        return selected
    }

    // MARK: - Inspection (for benchmarks & tests)

    public func neighbors(of internalID: Int32, at layer: Int) -> [Int32] {
        guard let n = node(for: internalID) else { return [] }
        return graphStorage.neighbors(of: n.vectorSlot, at: layer)
    }

    // MARK: - Private helpers

    /// Score the node at `internalID` against `query` using the configured metric.
    /// Re-fetches the pointer fresh from VectorStorage — never caches across appends.
    @inline(__always)
    func scored(_ internalID: Int32, _ query: UnsafePointer<Float>) -> Candidate {
        guard let n = node(for: internalID) else {
            // Should never happen; a missing node is a programming error.
            return Candidate(id: internalID, score: -.infinity)
        }
        let vec = vectorStorage.pointer(toSlot: Int(n.vectorSlot))
        let score = VectorMath.similarity(query, vec, dimension, metric: metric)
        return Candidate(id: internalID, score: score)
    }
}
