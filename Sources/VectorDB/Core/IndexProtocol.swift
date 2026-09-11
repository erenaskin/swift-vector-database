// IndexProtocol.swift
// Core/IndexProtocol.swift — Shared VectorIndex protocol that both FlatIndex
// and HNSWIndex conform to. VectorDB (the public actor) holds either
// implementation behind this single interface and switches transparently
// based on collection size (e.g. FlatIndex below ~2,000 vectors, HNSWIndex
// above), without the public API ever changing.

// MARK: - Architecture Overview (§4)
//
// ┌─────────────────────────────────────────────────────────────┐
// │  Public API (actor VectorDB)                                 │
// │  insert() / search() / delete() / save() / close()           │
// └───────────────┬─────────────────────────────────────────────┘
//                 │  String id <-> Int32 internal id  (IDMap)
// ┌───────────────▼─────────────────────────────────────────────┐
// │  Core Index  (FlatIndex  or  HNSWIndex)                      │
// │   - owns the graph / list logic                              │
// │   - calls into VectorMath for all distance calculations      │
// └───────────────┬─────────────────────┬────────────────────────┘
//                 │                     │
// ┌───────────────▼──────────────┐ ┌────▼────────────────────────┐
// │  VectorStorage                │ │  GraphStorage                │
// │  contiguous UnsafeMutablePtr  │ │  fixed-width adjacency lists │
// │  <Float>, row-major, N x D    │ │  per node per layer          │
// └───────────────┬──────────────┘ └────┬────────────────────────┘
//                 │                     │
// ┌───────────────▼─────────────────────▼────────────────────────┐
// │  PersistenceManager                                           │
// │   - MappedFile (mmap)  +  WriteAheadLog  +  snapshot/compact  │
// └───────────────────────────────────────────────────────────────┘
//
// Data flow for insert():
//   1. Public API validates dimension; copies [Float] into VectorStorage slot.
//   2. IDMap records String id → Int32 internal slot.
//   3. Core index (HNSW) runs insertion algorithm, writing neighbor lists
//      into GraphStorage.
//   4. PersistenceManager appends an insert record to the WAL (fast, durable).
//      The mmap'd snapshot file is NOT touched synchronously on every insert.
//   5. Periodically (or on save()), WAL is compacted into a fresh snapshot.
//
// Data flow for search():
//   1. Public API validates dimension; copies query into a temporary aligned
//      buffer.
//   2. Core index runs greedy graph descent (HNSW) using VectorMath distance
//      calls against VectorStorage.
//   3. Results (internal IDs + scores) are mapped back to String ids +
//      metadata via IDMap and returned as [SearchResult].

// MARK: - VectorIndex Protocol

/// The shared interface implemented by both FlatIndex (Phase 1, brute-force)
/// and HNSWIndex (Phase 4, approximate graph).
///
/// `VectorDB` (the public actor) holds an existential or generic reference to
/// this protocol and switches implementations based on collection size,
/// keeping the public API stable regardless of which engine is active.
///
/// All distance computation is delegated to `VectorMath` — neither index
/// implementation performs raw math itself.
///
/// - Note: All methods operate on raw `UnsafePointer<Float>` rather than
///   `[Float]` so that the call site in `VectorDB` copies the caller's array
///   into a temporary aligned buffer exactly once, and every downstream call
///   (across potentially millions of distance evaluations during search) pays
///   zero additional copy cost.
public protocol VectorIndex {
    /// The fixed vector dimensionality this index was created with.
    var dimension: Int { get }

    /// The number of vectors currently in the index (live, not tombstoned).
    var count: Int { get }

    /// Insert a vector at the given internal ID slot.
    /// - Parameters:
    ///   - internalID: The Int32 slot assigned by IDMap.
    ///   - vector:     Pointer to `dimension` Float values, already validated
    ///                 (no NaN/Inf) and normalized if metric is .cosine.
    mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws

    /// Remove the vector at the given internal ID.
    /// Phase 1 uses a naive mark-and-filter. Phase 7 (HNSW) uses tombstoning.
    mutating func remove(internalID: Int32) throws

    /// Return up to `k` (internalID, score) pairs, highest score first.
    /// Score convention: higher = more similar, for ALL metrics.
    /// `ef` is the HNSW candidate list size — ignored by FlatIndex.
    func search(query: UnsafePointer<Float>, k: Int, ef: Int?) -> [(id: Int32, score: Float)]

    /// Retrieve the raw vector for a given internal ID.
    /// - Returns: A copied array of floats, or nil if not found / tombstoned.
    func getVector(internalID: Int32) -> [Float]?
}
