/// IDMap.swift — Bidirectional String ↔ Int32 internal ID mapping.
///
/// IMPORTANT: Also stores the optional per-vector [String: String] metadata
/// bag (§3, §10.2, §12), which is returned alongside search results in
/// `SearchResult`.
///
/// Responsibilities:
///   - Assign monotonically increasing Int32 internal IDs to new external IDs.
///   - Enforce the duplicate-ID policy: insert() throws .duplicateID for
///     existing IDs; explicit update() is the only intentional-overwrite path.
///   - Provide O(1) bidirectional lookup.
///   - Store and retrieve per-vector metadata.
///   - Provide O(limit) pagination over live IDs in insertion order.
///
/// Thread safety: none — this struct is value-typed and must be guarded by
/// the caller's lock (the `VectorDatabase` actor owns exactly one instance).
///
/// `Sendable` is explicit and load-bearing: `VectorDatabase.save()` copies `idMap`
/// into a local `let` and hands that copy into a `Task.detached` closure so the
/// disk write doesn't block the actor. That capture crosses a concurrency
/// boundary. Every stored property is already safe by construction.

struct IDMap: Codable, Sendable {

    // MARK: - Storage

    private var stringToInt: [String: Int32] = [:]
    private var intToString: [Int32: String] = [:]

    /// Per-vector metadata bag. Stored here (not in VectorStorage) because it
    /// is variable-length text — unsuitable for the fixed-width float buffer.
    private var metadataStore: [Int32: [String: String]] = [:]

    /// Next unused internal ID. Monotonically increasing; never reused, even
    /// after deletion (tombstoned slots are reclaimed only during rebuild, §11).
    private var nextID: Int32 = 0

    /// Live internal IDs, kept sorted ascending — which, because IDs are handed
    /// out monotonically, is exactly insertion order.
    ///
    /// FIX O2 — WHY THIS ARRAY EXISTS:
    /// `listExternalIDs` used to walk `0..<nextID` and skip the gaps left by
    /// deletions. `nextID` never decreases, so in a long-lived store that has
    /// seen a million inserts and deleted most of them, `listIDs(offset: 0,
    /// limit: 10)` still had to iterate a million times to find ten live rows —
    /// pagination cost scaled with total historical inserts instead of with the
    /// page size. Maintaining the live set directly makes pagination
    /// O(offset + limit) bounded by the LIVE count, and in the common
    /// `offset == 0` case simply O(limit).
    ///
    /// Cost of maintenance: `assign` appends (amortised O(1), always at the end
    /// because IDs increase); `remove` does a binary search plus one `memmove`.
    /// A deletion already does strictly more work than that in the graph layer.
    private var liveIDs: [Int32] = []

    // MARK: - Codable

    private enum CodingKeys: String, CodingKey {
        case stringToInt, intToString, metadataStore, nextID, liveIDs
    }

    init() {}

    /// Custom decoding so that snapshots written by an earlier build — which had
    /// no `liveIDs` key — still load. The list is reconstructed from
    /// `intToString` in that case, which yields exactly the same content.
    /// `encode(to:)` stays compiler-synthesized and uses these same keys.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        stringToInt = try container.decode([String: Int32].self, forKey: .stringToInt)
        intToString = try container.decode([Int32: String].self, forKey: .intToString)
        metadataStore =
            try container.decodeIfPresent([Int32: [String: String]].self, forKey: .metadataStore)
            ?? [:]
        nextID = try container.decode(Int32.self, forKey: .nextID)

        if let stored = try container.decodeIfPresent([Int32].self, forKey: .liveIDs) {
            liveIDs = stored
        } else {
            liveIDs = intToString.keys.sorted()
        }
    }

    // MARK: - Counts

    /// Number of live (non-removed) mappings.
    var count: Int { stringToInt.count }

    /// Returns a page of live external IDs in insertion order.
    /// O(limit) work after an O(1) offset jump; O(limit) memory.
    func listExternalIDs(offset: Int, limit: Int) -> [String] {
        guard offset >= 0, limit > 0, offset < liveIDs.count else { return [] }

        let end = min(offset + limit, liveIDs.count)
        var result: [String] = []
        result.reserveCapacity(end - offset)
        for i in offset..<end {
            if let externalID = intToString[liveIDs[i]] {
                result.append(externalID)
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
            throw VectorDatabaseError.duplicateID(externalID)
        }
        // Overflow check: Int32.max ≈ 2.1 billion vectors — well beyond v1 scope,
        // but guard explicitly rather than silently wrapping.
        guard nextID < Int32.max else {
            throw VectorDatabaseError.invalidParameters(
                reason: "IDMap exhausted: reached Int32.max internal IDs"
            )
        }
        let id = nextID
        nextID += 1
        stringToInt[externalID] = id
        intToString[id] = externalID
        if let m = metadata { metadataStore[id] = m }
        insertLive(id)
        return id
    }

    /// Restores a mapping during WAL replay.
    /// This bypasses the duplicate ID check (WAL replay expects exact
    /// reconstruction) and ensures `nextID` is advanced past the restored
    /// `internalID` to prevent future collisions.
    mutating func restoreMapping(
        externalID: String,
        internalID: Int32,
        metadata: [String: String]? = nil
    ) {
        let alreadyLive = intToString[internalID] != nil

        stringToInt[externalID] = internalID
        intToString[internalID] = externalID
        if let m = metadata { metadataStore[internalID] = m }

        if !alreadyLive { insertLive(internalID) }

        // Advance nextID to ensure no future collisions
        nextID = max(nextID, internalID + 1)
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
            throw VectorDatabaseError.notFound(externalID)
        }
        stringToInt.removeValue(forKey: externalID)
        intToString.removeValue(forKey: id)
        metadataStore.removeValue(forKey: id)
        removeLive(id)
    }

    // MARK: - Update metadata

    /// Replaces the metadata for an already-registered ID.
    /// - Throws: `.notFound` if `externalID` is not registered.
    mutating func updateMetadata(for externalID: String, metadata: [String: String]?) throws {
        guard let id = stringToInt[externalID] else {
            throw VectorDatabaseError.notFound(externalID)
        }
        if let m = metadata {
            metadataStore[id] = m
        } else {
            metadataStore.removeValue(forKey: id)
        }
    }

    // MARK: - Live-ID list maintenance

    /// Inserts `id` while keeping `liveIDs` sorted ascending. The overwhelmingly
    /// common case (a fresh `assign`) appends to the end in O(1); WAL replay of
    /// an out-of-order ID falls back to a binary-search insert.
    private mutating func insertLive(_ id: Int32) {
        if let last = liveIDs.last, id <= last {
            let position = lowerBound(id)
            guard position == liveIDs.count || liveIDs[position] != id else { return }
            liveIDs.insert(id, at: position)
        } else {
            liveIDs.append(id)
        }
    }

    private mutating func removeLive(_ id: Int32) {
        let position = lowerBound(id)
        guard position < liveIDs.count, liveIDs[position] == id else { return }
        liveIDs.remove(at: position)
    }

    /// Index of the first element >= `id`.
    private func lowerBound(_ id: Int32) -> Int {
        var low = 0
        var high = liveIDs.count
        while low < high {
            let mid = low + (high - low) / 2
            if liveIDs[mid] < id {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }
}
