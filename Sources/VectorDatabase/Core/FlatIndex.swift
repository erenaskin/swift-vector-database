/// FlatIndex.swift — Brute-force exact index backed by VectorStorage.
///
/// Roles of FlatIndex:
///   (a) Correctness oracle for HNSW recall tests
///   (b) Transparent fallback inside VectorDatabase for small collections (<2k vectors)
///   (c) Re-ranking step after HNSW candidate retrieval
///
/// Pitfall: VectorStorage is a `final class` — if FlatIndex (a struct) is ever
/// copied, both copies share the same VectorStorage. This is intentional here
/// (same pattern as `[Float]`'s CoW internal buffer), but callers
/// should not rely on copy-on-write semantics for VectorStorage mutations.
/// The VectorDatabase public actor always owns exactly one FlatIndex.
///
/// Swap-remove (§7):
///   To remove slot S from N total slots:
///   1. Copy vector data from slot N-1 into slot S  (O(dim))
///   2. Update idToSlot/slotToID/squaredNorms for the moved ID (O(1))
///   3. Call storage.removeLast()                   (O(1))
///   Total: O(dim) — no holes, no tombstone filtering on every search.
/// Note: Marked `@unchecked Sendable` to allow advanced users parallel reads. Not internally thread-safe.
public struct FlatIndex: VectorIndex, @unchecked Sendable {

    // MARK: - Stored properties

    public let dimension: Int
    public let metric: DistanceMetric

    /// Row-major contiguous float buffer. Vector at slot i lives at
    /// `storage.pointer(toSlot: i)` — `i * dimension` floats into the buffer.
    private var storage: VectorStorage

    /// Maps internal Int32 ID → its current slot index in `storage`.
    /// Updated on every swap-remove to keep bookkeeping consistent.
    private var idToSlot: [Int32: Int] = [:]

    /// Maps slot index → the internal Int32 ID that currently occupies it.
    /// Parallel to `storage`; must stay in sync on every insert and remove.
    private var slotToID: [Int32] = []

    /// `‖vᵢ‖²` for the vector in each slot, parallel to `slotToID`.
    ///
    /// FIX O4 — WHY THIS CACHE EXISTS:
    /// The `.euclidean` search path used to call `vDSP_distancesq` once per
    /// stored vector, which is a per-row function call that defeats the whole
    /// point of having a contiguous row-major buffer. Caching one scalar per row
    /// lets `VectorMath.batchEuclideanSquared` compute every distance from a
    /// single cache-blocked `cblas_sgemv` call via
    /// `‖q-v‖² = ‖q‖² - 2(q·v) + ‖v‖²`, putting `.euclidean` on the same fast
    /// path `.dotProduct` and `.cosine` already enjoyed.
    ///
    /// Maintenance cost is one `vDSP_svesq` per insert and one array element
    /// copy per swap-remove — negligible next to the O(dim) data copy those
    /// operations already perform.
    private var squaredNorms: [Float] = []

    // MARK: - VectorIndex conformance

    /// Live vector count. With swap-remove there are no holes, so this is
    /// exactly `storage.count` (no tombstone subtraction needed).
    public var count: Int { storage.count }

    // MARK: - Initializer

    public init(dimension: Int, metric: DistanceMetric) {
        self.dimension = dimension
        self.metric = metric
        self.storage = VectorStorage(dimension: dimension)
    }

    // MARK: - Insert

    public mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
        let slot = storage.append(vector)
        idToSlot[internalID] = slot
        slotToID.append(internalID)
        squaredNorms.append(VectorMath.squaredNorm(vector, dimension))
    }

    // MARK: - Remove (real swap-remove backed by VectorStorage)

    public mutating func remove(internalID: Int32) throws {
        guard let slot = idToSlot[internalID] else {
            throw VectorDatabaseError.notFound("internalID \(internalID) not found in FlatIndex")
        }

        let lastSlot = storage.count - 1

        if slot != lastSlot {
            // Swap: overwrite the removed slot with the last slot's vector data.
            // This is O(dim) — safe because we re-fetch the pointer each time,
            // never holding a pointer across an append (no grow risk here since
            // we are only reading/writing existing slots, not appending).
            let src = storage.mutablePointer(toSlot: lastSlot)
            let dst = storage.mutablePointer(toSlot: slot)
            dst.update(from: src, count: dimension)

            // Update bookkeeping for the element that physically moved.
            let movedID = slotToID[lastSlot]
            slotToID[slot] = movedID
            idToSlot[movedID] = slot
            squaredNorms[slot] = squaredNorms[lastSlot]
        }

        // Release the last slot: remove its bookkeeping and shrink storage count.
        idToSlot.removeValue(forKey: internalID)
        slotToID.removeLast()
        squaredNorms.removeLast()
        storage.removeLast()
    }

    // MARK: - Search

    /// Returns up to `k` (internalID, score) pairs, highest score first.
    /// `ef` is ignored — meaningful only for HNSW.
    ///
    /// With swap-remove, every slot 0..<storage.count is live; no tombstone
    /// filtering is needed. The pointer is re-fetched per search so no
    /// dangling-pointer risk exists (no appends during search).
    public func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(
        id: Int32, score: Float
    )] {
        guard storage.count > 0, k > 0 else { return [] }

        let count = storage.count
        let buffer = storage.pointer(toSlot: 0)  // Contiguous access for batched operation

        let batchScores: [Float]
        switch metric {
        case .dotProduct:
            batchScores = VectorMath.batchDot(
                query: query, vectors: buffer, count: count, dim: dimension)
        case .cosine:
            batchScores = VectorMath.batchCosine(
                query: query, vectors: buffer, count: count, dim: dimension)
        case .euclidean:
            batchScores = squaredNorms.withUnsafeBufferPointer { norms in
                VectorMath.batchEuclideanSquared(
                    query: query, vectors: buffer, count: count, dim: dimension,
                    squaredNorms: norms.baseAddress!)
            }
        }

        // Use a min-heap by score (pops the LOWEST score) to track the top-k candidates.
        // This gives O(N log k) instead of sorting all N elements O(N log N).
        var topKHeap = BinaryHeap<Candidate>.minByScore()
        for i in 0..<count {
            let score = batchScores[i]
            if topKHeap.count < k {
                topKHeap.push(Candidate(id: slotToID[i], score: score))
            } else if let minCandidate = topKHeap.peek(), score > minCandidate.score {
                topKHeap.pop()
                topKHeap.push(Candidate(id: slotToID[i], score: score))
            }
        }

        // `topKHeap` is a minByScore heap, so draining it in priority order yields
        // worst-first; reverse for best-first.
        return topKHeap.drainedInPriorityOrder().reversed().map { (id: $0.id, score: $0.score) }
    }

    public func getVector(internalID: Int32) -> [Float]? {
        guard let slot = idToSlot[internalID] else { return nil }
        let ptr = storage.pointer(toSlot: slot)
        return Array(UnsafeBufferPointer(start: ptr, count: dimension))
    }

    // MARK: - Migration helper

    /// Re-inserts every vector currently in this FlatIndex into a fresh `HNSWIndex`.
    ///
    /// Called once by `IndexRouter` when the live count crosses `hnswThreshold`.
    /// Iterates `idToSlot` (private) without exposing it — callers stay decoupled
    /// from FlatIndex's internal storage layout.
    ///
    /// Insertion errors (e.g. duplicate IDs) are silently suppressed; this is safe
    /// because the FlatIndex already enforces uniqueness, so duplicates cannot occur.
    mutating func reinsertInto(hnsw: inout HNSWIndex) {
        for id in idToSlot.keys.sorted() {
            let slot = idToSlot[id]!
            let vec = storage.pointer(toSlot: slot)
            try? hnsw.insert(internalID: id, vector: vec)
        }
    }
}
