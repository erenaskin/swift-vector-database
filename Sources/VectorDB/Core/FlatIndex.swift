/// FlatIndex.swift — Phase 3 update: VectorStorage backing + real swap-remove.
///
/// Phase 1 used `[Float]` with a tombstone-based remove. Phase 3 replaces:
///   - `[Float] flatVectors` → `VectorStorage` (contiguous, grows geometrically)
///   - `tombstoned: Set<Int32>` → eliminated (swap-remove is O(1) and exact)
///
/// Roles of FlatIndex (unchanged from Phase 1):
///   (a) Correctness oracle for HNSW recall tests
///   (b) Transparent fallback inside VectorDB for small collections (<2k vectors)
///   (c) Re-ranking step after HNSW candidate retrieval
///
/// Pitfall: VectorStorage is a `final class` — if FlatIndex (a struct) is ever
/// copied, both copies share the same VectorStorage. This is intentional here
/// (same pattern as `[Float]` CoW internal buffer in Phase 1), but callers
/// should not rely on copy-on-write semantics for VectorStorage mutations.
/// The VectorDB public actor (Phase 8) always owns exactly one FlatIndex.
///
/// Swap-remove (§7):
///   To remove slot S from N total slots:
///   1. Copy vector data from slot N-1 into slot S  (O(dim))
///   2. Update idToSlot/slotToID for the moved ID   (O(1))
///   3. Call storage.removeLast()                   (O(1))
///   Total: O(dim) — no holes, no tombstone filtering on every search.
public struct FlatIndex: VectorIndex {

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

    // MARK: - VectorIndex conformance

    /// Live vector count. With swap-remove there are no holes, so this is
    /// exactly `storage.count` (no tombstone subtraction needed).
    public var count: Int { storage.count }

    // MARK: - Initializer

    public init(dimension: Int, metric: DistanceMetric) {
        self.dimension = dimension
        self.metric    = metric
        self.storage   = VectorStorage(dimension: dimension)
    }

    // MARK: - Insert

    public mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
        let slot = storage.append(vector)
        idToSlot[internalID] = slot
        slotToID.append(internalID)
    }

    // MARK: - Remove (Phase 3: real swap-remove backed by VectorStorage)

    public mutating func remove(internalID: Int32) throws {
        guard let slot = idToSlot[internalID] else {
            throw VectorDBError.notFound("internalID \(internalID) not found in FlatIndex")
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
            let movedID    = slotToID[lastSlot]
            slotToID[slot] = movedID
            idToSlot[movedID] = slot
        }

        // Release the last slot: remove its bookkeeping and shrink storage count.
        idToSlot.removeValue(forKey: internalID)
        slotToID.removeLast()
        storage.removeLast()
    }

    // MARK: - Search

    /// Returns up to `k` (internalID, score) pairs, highest score first.
    /// `ef` is ignored — meaningful only for HNSW (Phase 4).
    ///
    /// With swap-remove, every slot 0..<storage.count is live; no tombstone
    /// filtering is needed. The pointer is re-fetched per slot inside the loop
    /// so no dangling-pointer risk exists (no appends during search).
    public func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)] {
        guard storage.count > 0, k > 0 else { return [] }

        var scored: [(Int32, Float)] = []
        scored.reserveCapacity(storage.count)

        for i in 0..<storage.count {
            // Re-fetch on every iteration — safe because no mutation occurs
            // inside this loop (search is read-only).
            let score = VectorMath.similarity(
                query,
                storage.pointer(toSlot: i),
                dimension,
                metric: metric
            )
            scored.append((slotToID[i], score))
        }

        // Uniform descending sort: higher score = more similar, for all metrics.
        // Euclidean already returns -distSq so this single sort covers everything.
        scored.sort { $0.1 > $1.1 }
        return Array(scored.prefix(k))
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
        for (id, slot) in idToSlot {
            let vec = storage.pointer(toSlot: slot)
            try? hnsw.insert(internalID: id, vector: vec)
        }
    }
}
