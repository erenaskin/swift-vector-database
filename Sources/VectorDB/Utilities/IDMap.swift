/// IDMap.swift — Bidirectional String ↔ Int32 internal ID mapping.
///
/// IMPORTANT: Also stores the optional per-vector [String: String] metadata
/// bag (§3, §10.2, §12). This field MUST survive into Phase 8 and be
/// returned alongside search results in SearchResult. Do NOT design it away.
///
/// Responsibilities:
///   - Assign monotonically increasing Int32 internal IDs to new external IDs.
///   - Enforce the duplicate-ID policy: insert() throws .duplicateID for
///     existing IDs; explicit update() is the only intentional-overwrite path.
///   - Provide O(1) bidirectional lookup.
///   - Store and retrieve per-vector metadata.
///
/// Thread safety: none — this struct is value-typed and must be guarded by
/// the caller's write lock (Phase 5 ReadWriteLock).
struct IDMap: Codable {

    // MARK: - Storage

    private var stringToInt: [String: Int32] = [:]
    private var intToString: [Int32: String] = [:]

    /// Per-vector metadata bag. Stored here (not in VectorStorage) because it
    /// is variable-length text — unsuitable for the fixed-width float buffer.
    private var metadataStore: [Int32: [String: String]] = [:]

    /// Next unused internal ID. Monotonically increasing; never reused, even
    /// after deletion (tombstoned slots are reclaimed only during rebuild, §11).
    private var nextID: Int32 = 0

    // MARK: - VectorIndex conformance helper

    /// Number of live (non-removed) mappings.
    var count: Int { stringToInt.count }

    /// Returns a paginated list of live external IDs in insertion order.
    ///
    /// Trade-off: To avoid O(N log N) sorting and O(N) memory allocations,
    /// we iterate through the internal ID sequence (0..<nextID).
    /// This gives deterministic insertion order and takes O(offset + limit) time
    /// and O(limit) memory. For typical small-to-medium datasets, this is
    /// fast enough and strictly bounds memory usage.
    func listExternalIDs(offset: Int, limit: Int) -> [String] {
        guard offset >= 0, limit > 0, offset < count else { return [] }
        
        var result: [String] = []
        result.reserveCapacity(min(limit, count - offset))
        
        var liveSeen = 0
        // Iterate through all possible internal IDs in insertion order
        for id in 0..<nextID {
            if let externalID = intToString[id] {
                if liveSeen >= offset {
                    result.append(externalID)
                    if result.count == limit {
                        break
                    }
                }
                liveSeen += 1
            }
        }
        return result
    }

    // MARK: - ID Assignment

    /// Assigns a new monotonically increasing Int32 ID to `externalID`.
    /// - Throws: `.duplicateID` if `externalID` is already registered.
    /// - Returns: The newly assigned internal Int32 ID.
    mutating func assign(
        externalID: String,
        metadata: [String: String]? = nil
    ) throws -> Int32 {
        guard stringToInt[externalID] == nil else {
            throw VectorDBError.duplicateID(externalID)
        }
        let id = nextID
        // Overflow check: Int32.max ≈ 2.1 billion vectors — well beyond v1 scope,
        // but guard explicitly rather than silently wrapping.
        guard nextID < Int32.max else {
            throw VectorDBError.invalidParameters(
                reason: "IDMap exhausted: reached Int32.max internal IDs"
            )
        }
        nextID += 1
        stringToInt[externalID] = id
        intToString[id] = externalID
        if let m = metadata { metadataStore[id] = m }
        return id
    }

    // MARK: - Lookup

    /// Returns the internal ID for `externalID`, or nil if not registered.
    func internalID(for externalID: String) -> Int32? {
        stringToInt[externalID]
    }

    /// Returns the external string ID for `internalID`, or nil if not found.
    func externalID(for internalID: Int32) -> String? {
        intToString[internalID]
    }

    /// Returns the metadata bag for `internalID`, or nil if none was stored.
    func metadata(for internalID: Int32) -> [String: String]? {
        metadataStore[internalID]
    }

    // MARK: - Removal

    /// Removes the mapping for `externalID` (called during tombstone / rebuild).
    /// The internal Int32 ID is retired and never reissued.
    /// - Throws: `.notFound` if `externalID` is not registered.
    mutating func remove(externalID: String) throws {
        guard let id = stringToInt[externalID] else {
            throw VectorDBError.notFound(externalID)
        }
        stringToInt.removeValue(forKey: externalID)
        intToString.removeValue(forKey: id)
        metadataStore.removeValue(forKey: id)
    }

    // MARK: - Update metadata

    /// Replaces the metadata for an already-registered ID.
    /// - Throws: `.notFound` if `externalID` is not registered.
    mutating func updateMetadata(for externalID: String, metadata: [String: String]?) throws {
        guard let id = stringToInt[externalID] else {
            throw VectorDBError.notFound(externalID)
        }
        if let m = metadata {
            metadataStore[id] = m
        } else {
            metadataStore.removeValue(forKey: id)
        }
    }
}
