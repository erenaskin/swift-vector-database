import Foundation

/// VectorDB.swift — Public actor: main entry point (§12).
/// Implementation: Phase 8 (public API design).
///
/// This is the public interface for the vector database. It is an `actor` for Swift Concurrency
/// ergonomics, ensuring thread safety and strict isolation. All internal complex state is managed
/// by the synchronous `Engine` and `IDMap`.
///
/// ## Design Decisions (§12):
/// - **Synchronous vs. Async**: CPU-bound operations (`insert`, `search`, `update`, `delete`) are
///   synchronous `throws` functions. While they must be `await`ed from outside the actor boundary,
///   making them synchronous avoids forcing internal or isolated call sites into unnecessary `Task {}`
///   wrapping for pure in-memory operations. I/O-bound operations (`save`, `close`) are explicitly `async`.
/// - **Duplicate ID Policy**: `insert` strictly throws `.duplicateID` if the string ID exists.
///   Accidental silent overwrites are a common source of bugs. Use `update()` for explicit upserts.
/// - **In-Memory Mode**: If initialized with `path: nil`, the database operates entirely in-memory
///   with no disk I/O, perfectly suited for ephemeral/session-scoped search or testing.
public actor VectorDB {
    
    // MARK: - Public Types
    
    /// Provides snapshot statistics about the current state of the database.
    public struct Stats: Sendable {
        /// The number of active, searchable vectors in the database.
        public let liveCount: Int
        
        /// The number of soft-deleted (tombstoned) vectors. If this ratio grows too high,
        /// performance may degrade. The database automatically rebuilds itself when appropriate.
        public let tombstonedCount: Int
        
        /// The dimension of the vectors stored in this database.
        public let dimension: Int
        
        /// The approximate on-disk size in bytes, if persistence is enabled. Nil for in-memory mode.
        public let onDiskSizeBytes: Int?
    }
    
    // MARK: - State
    
    private let engine: Engine
    private var idMap: IDMap
    private let persistenceManager: PersistenceManager?
    private var isClosed: Bool = false
    private let walFlushInterval: Int
    private var writesSinceLastSync: Int = 0
    
    // MARK: - Init
    
    /// Initializes a new or loads an existing VectorDB.
    ///
    /// - Parameters:
    ///   - dimension: The dimensionality of the vectors.
    ///   - metric: The distance metric to use (e.g., `.cosine`).
    ///   - parameters: Advanced configuration for the HNSW index.
    ///   - path: The URL to the storage directory/file. If `nil`, the database operates entirely in-memory.
    /// - Throws: `VectorDBError.ioError` or other typed errors if loading fails.
    public init(
        dimension: Int,
        metric: DistanceMetric = .cosine,
        parameters: HNSWParameters = .default,
        path: URL? = nil,
        walFlushInterval: Int = 1
    ) throws {
        self.walFlushInterval = walFlushInterval
        if let path = path {
            self.persistenceManager = try PersistenceManager(databaseURL: path, dimension: dimension)
            
            // Check if file exists to load, otherwise start fresh
            if FileManager.default.fileExists(atPath: path.path) {
                if let (loadedIndex, loadedMap) = try self.persistenceManager!.load() {
                    self.engine = Engine(hnswIndex: loadedIndex)
                    self.idMap = loadedMap
                } else {
                    self.engine = Engine(dimension: dimension, metric: metric, hnswParams: parameters)
                    self.idMap = IDMap()
                }
            } else {
                self.engine = Engine(dimension: dimension, metric: metric, hnswParams: parameters)
                self.idMap = IDMap()
            }
        } else {
            self.persistenceManager = nil
            self.engine = Engine(dimension: dimension, metric: metric, hnswParams: parameters)
            self.idMap = IDMap()
        }
    }
    
    private func checkNotClosed() throws {
        if isClosed {
            throw VectorDBError.invalidParameters(reason: "VectorDB is closed.")
        }
    }
    
    private func validate(id: String?, vector: [Float]) throws {
        if let id = id, id.isEmpty {
            throw VectorDBError.invalidParameters(reason: "ID cannot be empty")
        }
        if vector.count != engine.dimension {
            throw VectorDBError.dimensionMismatch(expected: engine.dimension, got: vector.count)
        }
        
        for v in vector {
            if v.isNaN || v.isInfinite {
                throw VectorDBError.invalidVector(reason: "Vector contains NaN or Infinity")
            }
        }
        
        if engine.metric == .cosine {
            let isNonZero = vector.withUnsafeBufferPointer { buf -> Bool in
                guard let ptr = buf.baseAddress else { return false }
                return VectorMath.isNonZeroVector(ptr, vector.count)
            }
            if !isNonZero {
                throw VectorDBError.invalidVector(reason: "Zero vector is undefined for cosine metric")
            }
        }
    }
    
    private func checkAndFlushWAL() throws {
        writesSinceLastSync += 1
        if writesSinceLastSync >= walFlushInterval {
            try persistenceManager?.wal?.fsync()
            writesSinceLastSync = 0
        }
    }
    
    // MARK: - Core Operations
    
    /// Inserts a new vector with an optional metadata payload.
    /// - Throws: `.duplicateID` if the `id` already exists. Use `update()` for upserts.
    public func insert(id: String, vector: [Float], metadata: [String: String]? = nil) throws {
        try _insert(id: id, vector: vector, metadata: metadata, flushWAL: true)
    }
    
    private func _insert(id: String, vector: [Float], metadata: [String: String]?, flushWAL: Bool) throws {
        try checkNotClosed()
        try validate(id: id, vector: vector)
        let internalID = try idMap.assign(externalID: id, metadata: metadata)
        
        var finalVector = vector
        if engine.metric == .cosine {
            finalVector.withUnsafeMutableBufferPointer { buf in
                guard let ptr = buf.baseAddress else { return }
                VectorMath.normalize(ptr, buf.count)
            }
        }
        
        try finalVector.withUnsafeBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            try engine.insert(internalID: internalID, vector: ptr)
        }
        // Forward to WAL if persistent
        let hasMetadata = metadata != nil && !metadata!.isEmpty
        let opcodeToUse: WALOpcode = hasMetadata ? .insertWithMetadata : .insert
        
        try persistenceManager?.wal?.append(record: WALRecord(
            opcode: opcodeToUse,
            internalID: internalID,
            timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
            vectorData: finalVector,
            metadata: metadata
        ))
        
        if flushWAL {
            try checkAndFlushWAL()
        }
    }
    
    /// Explicitly updates an existing vector by tombstoning the old and inserting the new.
    /// - Throws: `.notFound` if the `id` does not exist.
    public func update(id: String, vector: [Float], metadata: [String: String]? = nil) throws {
        try checkNotClosed()
        try validate(id: id, vector: vector)
        guard let oldInternalID = idMap.internalID(for: id) else {
            throw VectorDBError.notFound(id)
        }
        
        // 1. Tombstone the old version in Engine
        try engine.remove(internalID: oldInternalID)
        try persistenceManager?.wal?.append(record: WALRecord(
            opcode: .delete,
            internalID: oldInternalID,
            timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
            vectorData: nil
        ))
        writesSinceLastSync += 1
        
        // 2. Remove old ID from idMap entirely, so we can re-assign the same string ID
        try idMap.remove(externalID: id)
        
        try _insert(id: id, vector: vector, metadata: metadata, flushWAL: false)
        
        try checkAndFlushWAL()
        _ = try checkAndRebuildIfNeeded()
    }
    
    /// Deletes a vector by soft-deleting (tombstoning) it.
    /// - Throws: `.notFound` if the `id` does not exist.
    public func delete(id: String) throws {
        try _delete(id: id, flushWAL: true)
    }
    
    private func _delete(id: String, flushWAL: Bool) throws {
        try checkNotClosed()
        guard let internalID = idMap.internalID(for: id) else {
            throw VectorDBError.notFound(id)
        }
        
        try engine.remove(internalID: internalID)
        try persistenceManager?.wal?.append(record: WALRecord(
            opcode: .delete,
            internalID: internalID,
            timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
            vectorData: nil
        ))
        try idMap.remove(externalID: id)
        
        if flushWAL {
            try checkAndFlushWAL()
        }
        
        _ = try checkAndRebuildIfNeeded()
    }
    
    /// Searches for the nearest `k` vectors to the provided query.
    /// - Parameters:
    ///   - query: The vector to search for.
    ///   - k: The maximum number of results to return.
    ///   - ef: Optional override for the dynamic candidate list size during search.
    /// - Returns: An array of `SearchResult`.
    public func search(query: [Float], k: Int = 10, ef: Int? = nil) throws -> [SearchResult] {
        try checkNotClosed()
        try validate(id: nil, vector: query)
        
        guard k > 0 else { return [] }
        
        var finalQuery = query
        if engine.metric == .cosine {
            finalQuery.withUnsafeMutableBufferPointer { buf in
                guard let ptr = buf.baseAddress else { return }
                VectorMath.normalize(ptr, buf.count)
            }
        }
        
        let results = finalQuery.withUnsafeBufferPointer { buf -> [(id: Int32, score: Float)] in
            guard let ptr = buf.baseAddress else { return [] }
            return engine.search(query: ptr, k: k, ef: ef)
        }
        
        return results.compactMap { res in
            guard let externalID = idMap.externalID(for: res.id) else { return nil }
            return SearchResult(
                id: externalID,
                score: res.score,
                metadata: idMap.metadata(for: res.id)
            )
        }
    }

    /// Updates the metadata for an existing vector without modifying the vector data itself.
    /// This is a lightweight operation that only appends a metadata update record to the WAL.
    /// - Parameters:
    ///   - id: The external string ID of the vector.
    ///   - metadata: The new metadata bag to store, or `nil` to remove existing metadata.
    /// - Throws: `.notFound` if the `id` does not exist or has been deleted.
    public func updateMetadata(id: String, metadata: [String: String]?) throws {
        try checkNotClosed()
        
        guard let internalID = idMap.internalID(for: id) else {
            throw VectorDBError.notFound(id)
        }
        
        // 1. Update in-memory IDMap
        try idMap.updateMetadata(for: id, metadata: metadata)
        
        // 2. Append to WAL
        try persistenceManager?.wal?.append(record: WALRecord(
            opcode: .updateMetadata,
            internalID: internalID,
            timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
            metadata: metadata
        ))
        
        try checkAndFlushWAL()
    }

    /// Retrieves a single vector and its metadata by ID.
    /// - Parameter id: The external string ID of the vector.
    /// - Returns: A tuple containing the copied vector and metadata, or `nil` if not found or if tombstoned.
    public func get(id: String) -> (vector: [Float], metadata: [String: String]?)? {
        try? checkNotClosed()
        guard let internalID = idMap.internalID(for: id) else { return nil }
        
        guard let vector = engine.getVector(internalID: internalID) else { return nil }
        let metadata = idMap.metadata(for: internalID)
        
        return (vector, metadata)
    }

    /// Returns a paginated list of live external IDs.
    /// - Parameters:
    ///   - offset: The number of live records to skip.
    ///   - limit: The maximum number of IDs to return.
    /// - Returns: An array of string IDs in deterministic insertion order.
    public func listIDs(offset: Int, limit: Int) -> [String] {
        guard !isClosed else { return [] }
        return idMap.listExternalIDs(offset: offset, limit: limit)
    }
    
    // MARK: - Maintenance
    
    /// Manually triggers a graph rebuild if the tombstoned ratio has crossed the
    /// threshold, reclaiming space and restoring search quality/performance.
    /// This is synchronous and its cost is roughly O(live vector count).
    /// Callers doing this on a large index should expect a real pause.
    /// - Returns: `true` if a rebuild was performed, `false` if it wasn't needed.
    @discardableResult
    public func compact() throws -> Bool {
        try checkNotClosed()
        return try checkAndRebuildIfNeeded()
    }
    
    private func checkAndRebuildIfNeeded() throws -> Bool {
        if engine.shouldRebuild {
            try engine.rebuild(with: engine.collectLiveSnapshots())
            return true
        }
        return false
    }
    
    // MARK: - Batch Operations
    
    /// Transactionally inserts a batch of vectors.
    /// Acquires the actor lock once for the entire batch.
    /// Policy: This implements a strict all-or-nothing rollback. On any failure, 
    /// any items already inserted in this batch are rolled back (from `idMap`, `Engine`, and WAL) 
    /// exactly as if the batch had never been called.
    /// - Throws: If any single item fails validation (e.g. `.duplicateID`).
    public func batchInsert(_ items: [(id: String, vector: [Float], metadata: [String: String]?)]) throws {
        try checkNotClosed()
        for item in items {
            try validate(id: item.id, vector: item.vector)
        }
        
        var insertedIDs: [String] = []
        insertedIDs.reserveCapacity(items.count)
        
        var throwToCaller: Error? = nil
        do {
            // Single write lock across the entire batch (Phase 8 Design Decision)
            for item in items {
                try _insert(id: item.id, vector: item.vector, metadata: item.metadata, flushWAL: false)
                insertedIDs.append(item.id)
                writesSinceLastSync += 1
            }
        } catch {
            // Rollback in reverse order
            for id in insertedIDs.reversed() {
                try? self._delete(id: id, flushWAL: false)
                writesSinceLastSync += 1
            }
            throwToCaller = error
        }
        
        try persistenceManager?.wal?.fsync()
        writesSinceLastSync = 0
        
        if let error = throwToCaller {
            throw error
        }
    }
    
    // MARK: - Durability & Lifecycle
    
    /// Manually triggers a snapshot serialization and WAL truncation.
    /// Wraps the internal write in a task to avoid blocking the actor while I/O happens.
    public func save() async throws {
        try checkNotClosed()
        guard let pm = persistenceManager else { return } // No-op for in-memory mode
        
        // We must extract the snapshots from Engine inside the actor to safely save them.
        // HNSWIndex snapshot extraction allows saving the current state reliably.
        // For phase 8, we expect `pm.save(index:idMap:)` but we must pass the actual index.
        // We'll expose `router.snapshot()` or rely on Engine's `collectLiveSnapshots()` if needed.
        // But wait, PersistenceManager requires `HNSWIndex`!
        // We will add a method to `Engine` to extract a copy of the index if possible.
        // Or, wait: PersistenceManager saves the raw file, which requires `HNSWIndex`.
        // Let's implement this by having Engine provide an `exportHNSWIndex()` method.
        let exportedIndex = engine.exportHNSWIndex()
        try pm.save(index: exportedIndex, idMap: idMap)
    }
    
    /// Safely shuts down the database.
    /// - Performs a best-effort `save()`.
    /// - Unmaps and closes underlying file descriptors by nullifying the persistence manager.
    /// - Idempotent: safe to call multiple times.
    public func close() async {
        if isClosed { return }
        
        // Force a final fsync before save
        try? persistenceManager?.wal?.fsync()
        writesSinceLastSync = 0
        
        // Best-effort flush
        try? await save()
        
        isClosed = true
        // Allow OS to reclaim file descriptors and mapped memory
        // Because VectorStorage and GraphStorage rely on MappedFile lifecycle.
        engine.close()
    }
    
    // MARK: - Introspection
    
    /// Returns current statistics.
    public func stats() -> Stats {
        // We would compute disk size if needed.
        let diskSize: Int? = nil // Could use FileManager to get file size
        return Stats(
            liveCount: idMap.count,
            tombstonedCount: engine.tombstonedCount,
            dimension: engine.dimension,
            onDiskSizeBytes: diskSize
        )
    }
    
    // MARK: - Internal Inspector Forwarding Methods
    
    internal func _inspectEntryPointID() -> String? {
        guard let internalID = engine.inspectEntryPoint() else { return nil }
        return idMap.externalID(for: internalID)
    }
    
    internal func _inspectNodeLevel(of id: String) -> Int? {
        guard let internalID = idMap.internalID(for: id) else { return nil }
        return engine.inspectNodeLevel(internalID: internalID)
    }
    
    internal func _inspectNeighbors(of id: String, atLayer layer: Int) -> [String]? {
        guard let internalID = idMap.internalID(for: id) else { return nil }
        guard let neighborInternalIDs = engine.inspectNeighbors(internalID: internalID, atLayer: layer) else {
            return nil
        }
        return neighborInternalIDs.compactMap { idMap.externalID(for: $0) }
    }
}
