/// HNSWIndex.swift — Phase 4 approximate nearest-neighbor graph engine (§8).
///
/// Implements §8.4 (randomLevel), §8.5 (searchLayer), §8.6 (insert +
/// selectNeighborsHeuristic), and §8.7 (search/query).
///
/// ARCHITECTURE:
///   - `vectorStorage: VectorStorage` — raw float data (Phase 3)
///   - `graphStorage: GraphStorage`   — fixed-width adjacency lists (Phase 4a)
///   - `nodes: [Int32: HNSWNode]`     — maps internalID → (level, vectorSlot)
///   - `rng: SeedableRNG`             — seeded PRNG for reproducible level assignment
///
/// SEEDED RNG (§8.4 / §8 Pitfalls):
///   "does my recall regression test still pass" is meaningless if the graph
///   structure changes on every run. HNSWIndex always uses `SeedableRNG` — the
///   same xorshift64 used in TestFixtures — seeded from `params.seed`. The seed
///   is part of the serialised format (Phase 6) so the graph can be reproduced
///   identically after a reload.
///
/// HEAP CONVENTION (§8.5 Pitfall):
///   This codebase uses "higher score = more similar" throughout. The HNSW paper
///   uses "smaller distance = closer". The two heaps in `searchLayer` are:
///     · candidateHeap: pop HIGHEST score first  → explore closest candidates first
///     · foundHeap:     pop LOWEST  score first  → evict FARTHEST when set is full

import Foundation

public struct HNSWIndex: VectorIndex {

    // MARK: - Stored properties

    public let dimension: Int
    public let metric: DistanceMetric
    public let params: HNSWParameters

    internal var vectorStorage: VectorStorage
    internal var graphStorage: GraphStorage

    /// Maps internalID → node metadata.
    var nodes: [Int32: HNSWNode] = [:]

    /// Tracks soft-deleted internal IDs (§11).
    var tombstoned: Set<Int32> = []

    /// Current global entry point (the node at the topmost layer).
    var entryPoint: Int32? = nil
    var entryPointLevel: Int = 0

    /// Seeded RNG for deterministic level assignment (§8.4).
    /// MUST NOT be replaced with the global `Double.random` — see file header.
    public var rng: SeedableRNG

    // MARK: - VectorIndex conformance

    public var count: Int { vectorStorage.count }

    // MARK: - Init

    public init(dimension: Int, metric: DistanceMetric, params: HNSWParameters = .default) {
        self.dimension = dimension
        self.metric    = metric
        self.params    = params
        self.vectorStorage = VectorStorage(dimension: dimension)
        self.graphStorage  = GraphStorage(m: params.M, mMax0: params.Mmax0)
        self.rng = SeedableRNG(seed: params.seed)
    }

    // MARK: - VectorIndex: insert (§8.6)
    //
    // INSERT PITFALLS ADDRESSED (§8 Pitfalls):
    //
    // [Pitfall-1] Entry-point level must stay correct.
    //   The entry point is updated to the new node ONLY AFTER Phase B completes
    //   (line: `if level > entryPointLevel { entryPoint = internalID ... }`).  This
    //   means every in-progress insert observes the old entry point, which is stable
    //   because Phase 5 serialises inserts under a write lock. Updating before Phase B
    //   would allow a concurrent reader to traverse from an incompletely-wired node.
    //
    // [Pitfall-2] Recursive lock reentrancy during neighbour pruning.
    //   `selectNeighborsHeuristic` is called from inside Phase B to prune OTHER
    //   nodes' neighbour lists (bidirectional wiring).  This means a single top-level
    //   `insert` call mutates the graph structure of multiple nodes.
    //   • Phase 5 wraps the entire `insert` call in a single write lock acquisition.
    //   • `graphStorage.setNeighbors` (called during pruning) does NOT attempt to
    //     reacquire the lock — it is purely an internal struct mutation.
    //   • A non-reentrant pthread_rwlock would deadlock if the pruning step tried to
    //     acquire it again. Our design avoids this by keeping the entire insert — both
    //     the primary wiring and the pruning — inside one critical section.
    // We assume the caller (Engine) has acquired an exclusive write lock.

    public mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
        let slot  = vectorStorage.append(vector)
        let level = randomLevel()

        guard let entryID = entryPoint else {
            // First node: becomes entry point at its own level, no edges.
            entryPoint      = internalID
            entryPointLevel = level
            nodes[internalID] = HNSWNode(level: level, vectorSlot: Int32(slot))
            graphStorage.addNode()
            return
        }

        nodes[internalID] = HNSWNode(level: level, vectorSlot: Int32(slot))
        graphStorage.addNode()

        var currentNearest = [entryID]

        // Phase A: descend from top layer to just above the new node's level,
        // greedy ef=1 search at each layer to find a good entry point.
        if level < entryPointLevel {
            for lc in stride(from: entryPointLevel, to: level, by: -1) {
                currentNearest = searchLayer(query: vector,
                                             entryPoints: currentNearest,
                                             ef: 1,
                                             layer: lc).map(\.id)
            }
        }

        // Phase B: from min(entryPointLevel, level) down to 0, full efConstruction
        // search, neighbor selection, bidirectional wiring, and pruning.
        for lc in stride(from: min(entryPointLevel, level), through: 0, by: -1) {
            let candidates = searchLayer(query: vector,
                                         entryPoints: currentNearest,
                                         ef: params.efConstruction,
                                         layer: lc)
            
            let maxConn = (lc == 0) ? params.Mmax0 : params.M
            let selected = selectNeighborsHeuristic(candidates: candidates, m: maxConn)

            graphStorage.setNeighbors(of: Int32(slot), at: lc, to: selected.map(\.id))

            for neighbor in selected {
                let neighborMax = (lc == 0) ? params.Mmax0 : params.M
                guard let neighborNode = nodes[neighbor.id] else { continue }
                let currentNeighborCount = graphStorage.neighborCount(of: neighborNode.vectorSlot, at: lc)

                if currentNeighborCount < neighborMax {
                    graphStorage.addNeighbor(of: neighborNode.vectorSlot, at: lc, neighborID: internalID)
                } else {
                    // Node is full. We cannot just addNeighbor because GraphStorage is fixed-width.
                    // Instead, gather existing neighbors + the new one, prune, and overwrite.
                    let neighborVec = vectorStorage.pointer(toSlot: Int(neighborNode.vectorSlot))
                    
                    var pool = graphStorage.neighbors(of: neighborNode.vectorSlot, at: lc)
                        .map { scored($0, neighborVec) }
                    
                    pool.append(scored(internalID, neighborVec))
                    
                    let pruned = selectNeighborsHeuristic(candidates: pool, m: neighborMax)
                    graphStorage.setNeighbors(of: neighborNode.vectorSlot, at: lc, to: pruned.map(\.id))
                }
            }

            currentNearest = candidates.map(\.id)
        }

        if level > entryPointLevel {
            entryPoint      = internalID
            entryPointLevel = level
        }
    }

    // MARK: - VectorIndex: remove

    public mutating func remove(internalID: Int32) throws {
        guard nodes[internalID] != nil, !tombstoned.contains(internalID) else {
            throw VectorDBError.notFound("internalID \(internalID) not found in HNSWIndex")
        }
        
        tombstoned.insert(internalID)
        
        // Pitfall 11.2: correct entry-point reassignment when the current entry point gets tombstoned.
        // We pick a new one from the highest surviving non-tombstoned level,
        // without removing the old node from the graph structure itself.
        if internalID == entryPoint {
            var newEntryPoint: Int32? = nil
            var maxLevel = -1
            
            for (id, node) in nodes {
                if !tombstoned.contains(id), node.level > maxLevel {
                    maxLevel = node.level
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

    // MARK: - VectorIndex: search (§8.7)

    public func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)] {
        guard let entryID = entryPoint, k > 0 else { return [] }
        
        // §11: fetch `k + tombstoned.count`, then filter, then take `k`.
        let efSearch = max(ef ?? params.efSearch, k + tombstoned.count)

        var currentNearest = [entryID]
        for lc in stride(from: entryPointLevel, to: 0, by: -1) {
            currentNearest = searchLayer(query: query,
                                         entryPoints: currentNearest,
                                         ef: 1,
                                         layer: lc).map(\.id)
        }
        let results = searchLayer(query: query,
                                  entryPoints: currentNearest,
                                  ef: efSearch,
                                  layer: 0)
        
        // §11: filter out tombstoned nodes, then take k
        let liveResults = results.filter { !tombstoned.contains($0.id) }
        return Array(liveResults.prefix(k)).map { ($0.id, $0.score) }
    }

    public func getVector(internalID: Int32) -> [Float]? {
        guard let node = nodes[internalID], !tombstoned.contains(internalID) else { return nil }
        let ptr = vectorStorage.pointer(toSlot: Int(node.vectorSlot))
        return Array(UnsafeBufferPointer(start: ptr, count: dimension))
    }

    // MARK: - §8.4 Level Assignment

    /// Malkov & Yashunin's exponential decay level assignment (§8.4).
    ///
    /// Uses the instance-level `rng` (SeedableRNG / xorshift64), NOT the global
    /// `Double.random`. This is mandatory: deterministic graph construction is
    /// required for reproducible recall regression tests (§8 Pitfalls, §8 DoD).
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
    /// Both heaps operate in "higher score = more similar" space. The PAPER calls
    /// these "min-heap by distance" and "max-heap by distance" respectively,
    /// which is INVERTED from our score space. Unit tests in HeapOrderingTests
    /// verify this before anything builds on top.
    func searchLayer(query: UnsafePointer<Float>,
                     entryPoints: [Int32],
                     ef: Int,
                     layer: Int) -> [Candidate] {
        guard !entryPoints.isEmpty else { return [] }

        var visited = Set<Int32>(entryPoints)

        // Seed both heaps from the entry points.
        let seedCandidates = entryPoints.map { scored($0, query) }
        // candidateHeap: pop CLOSEST (highest score) next — "min-heap by distance" in paper.
        var candidateHeap = BinaryHeap<Candidate>.maxByScore(seedCandidates)
        // foundHeap: peek/pop FARTHEST (lowest score) — "max-heap by distance" in paper.
        var foundHeap     = BinaryHeap<Candidate>.minByScore(seedCandidates)

        while let c = candidateHeap.pop() {
            guard let worst = foundHeap.peek() else { break }
            // Early exit: the closest unexplored candidate is farther than our
            // worst result and the result set is already full.
            if c.score < worst.score && foundHeap.count >= ef { break }

            guard let cNode = nodes[c.id] else { continue }
            for neighborID in graphStorage.neighbors(of: cNode.vectorSlot, at: layer) {
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

        // foundHeap (minByScore) pops the worst (lowest) score first.
        // We pop everything and reverse to return highest-score first.
        return foundHeap.sortedDescending().reversed()
    }

    // MARK: - §8.6 Neighbor Selection Heuristic

    /// Diversity-favoring neighbor selection (§8.6). Prefers candidates that are
    /// closer to the query than to any already-selected neighbor. This produces
    /// better-connected graphs than naive top-m selection.
    ///
    /// Backfill: if diversity pruning leaves fewer than `m` selected, fills with
    /// the next-closest remaining candidates.
    private func selectNeighborsHeuristic(candidates: [Candidate], m: Int) -> [Candidate] {
        var sorted   = candidates.sorted { $0.score > $1.score } // closest first
        var selected: [Candidate] = []

        while let candidate = sorted.first, selected.count < m {
            sorted.removeFirst()
            guard let candidateNode = nodes[candidate.id] else { continue }
            let candidateVec = vectorStorage.pointer(toSlot: Int(candidateNode.vectorSlot))

            let isDiverse = selected.allSatisfy { existing in
                guard let existingNode = nodes[existing.id] else { return true }
                let existingVec = vectorStorage.pointer(toSlot: Int(existingNode.vectorSlot))
                let distToExisting = VectorMath.similarity(candidateVec, existingVec,
                                                           dimension, metric: metric)
                return candidate.score > distToExisting
            }

            if isDiverse || selected.isEmpty {
                selected.append(candidate)
            }
        }

        // Backfill if diversity pruning left us short.
        if selected.count < m {
            for candidate in candidates.sorted(by: { $0.score > $1.score }) {
                if selected.count >= m { break }
                if !selected.contains(where: { $0.id == candidate.id }) {
                    selected.append(candidate)
                }
            }
        }
        return selected
    }

    // MARK: - Inspection (for benchmarks & tests)

    public func neighbors(of internalID: Int32, at layer: Int) -> [Int32] {
        guard let node = nodes[internalID] else { return [] }
        return graphStorage.neighbors(of: node.vectorSlot, at: layer)
    }

    // MARK: - Private helpers

    /// Score the node at `internalID` against `query` using the configured metric.
    /// Re-fetches the pointer fresh from VectorStorage — never caches across appends.
    func scored(_ internalID: Int32, _ query: UnsafePointer<Float>) -> Candidate {
        guard let node = nodes[internalID] else {
            // Should never happen; a missing node is a programming error.
            return Candidate(id: internalID, score: -.infinity)
        }
        let vec   = vectorStorage.pointer(toSlot: Int(node.vectorSlot))
        let score = VectorMath.similarity(query, vec, dimension, metric: metric)
        return Candidate(id: internalID, score: score)
    }
}
