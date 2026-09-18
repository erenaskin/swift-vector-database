import Foundation

/// VectorDatabase.swift — Public actor: main entry point (§12).
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
///
/// ## Actor reentrancy, and why it matters here
/// Every `await` inside this actor is a point where OTHER calls can interleave. Two bugs came
/// directly from ignoring that (see `save()` and `checkAndRebuildIfNeeded()`), so the rule this
/// file now follows is explicit: any operation that reads state, does work off-actor, and then
/// writes state back must either hold no such state across the `await`, or must reconcile
/// whatever happened during it. There is no third option.
public actor VectorDatabase {

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

        /// The combined on-disk size of the snapshot and WAL files in bytes, if
        /// persistence is enabled. `nil` for in-memory mode, or before anything
        /// has been written.
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

    /// Initializes a new or loads an existing VectorDatabase.
    ///
    /// - Parameters:
    ///   - dimension: The dimensionality of the vectors.
    ///   - metric: The distance metric to use (e.g., `.cosine`).
    ///   - parameters: Advanced configuration for the HNSW index.
    ///   - path: The URL of the database FILE (not a directory). Sibling files
    ///     `<path>.wal`, `<path>.lock` and `<path>.tmp` are created next to it.
    ///     If `nil`, the database operates entirely in-memory.
    ///   - walFlushInterval: How many writes to buffer before calling `fsync` on
    ///     the WAL. `1` (the default) fsyncs on every write: maximum durability,
    ///     minimum throughput.
    /// - Throws: `.invalidParameters` for a bad `parameters` value,
    ///   `.dimensionMismatch` if an existing file at `path` was written with a
    ///   different dimension, or `.ioError`/`.corruptFile` if loading fails.
    public init(
        dimension: Int,
        metric: DistanceMetric = .cosine,
        parameters: HNSWParameters = .default,
        path: URL? = nil,
        walFlushInterval: Int = 1
    ) throws {
        // Fix Y4: parameters used to be validated inside `HNSWParameters.init`
        // with `fatalError`, which crashed the host app on a bad value. They are
        // validated here instead, as a normal throwing error, because this
        // initializer is already `throws` and callers can actually handle it.
        try parameters.validate()

        guard dimension > 0 else {
            throw VectorDatabaseError.invalidParameters(
                reason: "dimension must be greater than 0 (got \(dimension)).")
        }
        guard walFlushInterval >= 1 else {
            throw VectorDatabaseError.invalidParameters(
                reason: "walFlushInterval must be at least 1 (got \(walFlushInterval)).")
        }

        self.walFlushInterval = walFlushInterval
        if let path = path {
            let manager = try PersistenceManager(databaseURL: path, dimension: dimension)
            self.persistenceManager = manager

            // Check if a snapshot/WAL exists to load, otherwise start fresh
            if let (loadedIndex, loadedMap) = try manager.load(
                fallbackMetric: metric, fallbackParams: parameters)
            {
                self.engine = Engine(hnswIndex: loadedIndex)
                self.idMap = loadedMap
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
            throw VectorDatabaseError.invalidParameters(reason: "VectorDatabase is closed.")
        }
    }

    private func validate(id: String?, vector: [Float]) throws {
        if let id = id, id.isEmpty {
            throw VectorDatabaseError.invalidParameters(reason: "ID cannot be empty")
        }
        if vector.count != engine.dimension {
            throw VectorDatabaseError.dimensionMismatch(
                expected: engine.dimension, got: vector.count)
        }

        try vector.withUnsafeBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }

            // Fix S2: use the shared VectorMath helper instead of an open-coded
            // duplicate loop. `isFiniteVector` existed for exactly this call site
            // but nothing in Sources was using it.
            if !VectorMath.isFiniteVector(ptr, buf.count) {
                throw VectorDatabaseError.invalidVector(reason: "Vector contains NaN or Infinity")
            }

            if engine.metric == .cosine, !VectorMath.isNonZeroVector(ptr, buf.count) {
                throw VectorDatabaseError.invalidVector(
                    reason: "Zero vector is undefined for cosine metric")
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

    private func _insert(id: String, vector: [Float], metadata: [String: String]?, flushWAL: Bool)
        throws
    {
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

        try persistenceManager?.wal?.append(
            record: WALRecord(
                opcode: opcodeToUse,
                internalID: internalID,
                externalID: id,
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
            throw VectorDatabaseError.notFound(id)
        }

        // 1. Tombstone the old version in Engine
        try engine.remove(internalID: oldInternalID)
        try persistenceManager?.wal?.append(
            record: WALRecord(
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
        try rebuildIfNeeded()
    }

    /// Deletes a vector by soft-deleting (tombstoning) it.
    /// - Throws: `.notFound` if the `id` does not exist.
    public func delete(id: String) throws {
        try _delete(id: id, flushWAL: true)
    }

    private func _delete(id: String, flushWAL: Bool) throws {
        try checkNotClosed()
        guard let internalID = idMap.internalID(for: id) else {
            throw VectorDatabaseError.notFound(id)
        }

        try engine.remove(internalID: internalID)
        try persistenceManager?.wal?.append(
            record: WALRecord(
                opcode: .delete,
                internalID: internalID,
                timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
                vectorData: nil
            ))
        try idMap.remove(externalID: id)

        if flushWAL {
            try checkAndFlushWAL()
        }

        try rebuildIfNeeded()
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

    /// Runs several independent queries in parallel and returns their results in the
    /// same order as `queries`.
    ///
    /// `Engine` is guarded by a `ReadWriteLock` that allows many simultaneous
    /// READERS. Because `VectorDatabase` is an `actor`, two SEPARATE calls to
    /// `search(query:)` from outside never actually run concurrently — the actor
    /// serializes them — so that multi-reader capability has no way to be
    /// exercised through the single-query API. `searchBatch` fixes this: it is
    /// still one synchronous, actor-isolated call (no `await` inside, so actor
    /// reentrancy is not a concern), but internally it fans the N queries out
    /// across multiple OS threads via `DispatchQueue.concurrentPerform`, each of
    /// which calls `engine.search(...)` and genuinely acquires the shared read
    /// lock at the same time as the others.
    ///
    /// - Parameters:
    ///   - queries: The vectors to search for. Order is preserved in the result.
    ///   - k: The maximum number of results to return per query.
    ///   - ef: Optional override for the dynamic candidate list size during search.
    /// - Returns: One `[SearchResult]` array per input query, in the same order.
    public func searchBatch(queries: [[Float]], k: Int = 10, ef: Int? = nil) throws
        -> [[SearchResult]]
    {
        try checkNotClosed()
        guard !queries.isEmpty else { return [] }
        for query in queries {
            try validate(id: nil, vector: query)
        }
        guard k > 0 else { return Array(repeating: [], count: queries.count) }

        var normalizedQueries = queries
        if engine.metric == .cosine {
            for i in normalizedQueries.indices {
                normalizedQueries[i].withUnsafeMutableBufferPointer { buf in
                    guard let ptr = buf.baseAddress else { return }
                    VectorMath.normalize(ptr, buf.count)
                }
            }
        }

        // FIX Y1 (round 2) — THE UNSAFE POINTER NEVER TOUCHES ACTOR ISOLATION AT ALL.
        //
        // The first attempt at this fix kept the pointer-fan-out logic inline in
        // this actor-isolated method, with the pointer split across two bindings
        // (`resultsBase` inside `concurrentPerform`, `rawResults` in the final
        // `.map`). Swift 6's region-based isolation checker treats a value used
        // inside a closure handed to a concurrent API (`concurrentPerform`) as
        // "sent" into a separate region; any further use of anything aliasing
        // that same allocation from actor-isolated code — even under a different
        // local name — is then flagged as a potential race, regardless of the
        // fact that `concurrentPerform` has actually returned by that point.
        // `nonisolated(unsafe)` on only ONE of the two aliases didn't cover the
        // other.
        //
        // Rather than fight the checker with more annotations, the entire
        // pointer dance is moved into `runParallelSearches`, a `nonisolated`
        // method below. Nonisolated code is not part of the actor's isolation
        // domain in the first place, so there is no "actor-isolated closure"
        // for the checker to reason about — this is ordinary synchronous code
        // that happens to use threads internally, exactly like calling into a
        // C library would be. It touches only its parameters (`engine`,
        // `queries`, `k`, `ef`), never `self.idMap` or any other isolated state,
        // which is what makes moving it off the actor sound.
        let queriesToRun = normalizedQueries
        let capturedEngine = engine
        let rawResults = runParallelSearches(
            engine: capturedEngine, queries: queriesToRun, k: k, ef: ef)

        return rawResults.map { perQuery in
            perQuery.compactMap { res in
                guard let externalID = idMap.externalID(for: res.id) else { return nil }
                return SearchResult(
                    id: externalID,
                    score: res.score,
                    metadata: idMap.metadata(for: res.id)
                )
            }
        }
    }

    /// Runs `queries.count` independent searches against `engine` across
    /// multiple OS threads, returning raw (internalID, score) pairs per query in
    /// the same order as `queries`.
    ///
    /// Deliberately `nonisolated`: this function's only inputs are its
    /// parameters, and `Engine` already does its own internal locking
    /// (`ReadWriteLock`), so nothing here needs — or should have — access to
    /// `VectorDatabase`'s actor-isolated state.
    private nonisolated func runParallelSearches(
        engine: Engine,
        queries: [[Float]],
        k: Int,
        ef: Int?
    ) -> [[(id: Int32, score: Float)]] {
        let count = queries.count
        var results = [[(id: Int32, score: Float)]](repeating: [], count: count)

        // All use of the raw buffer pointer — writing from worker threads during
        // `concurrentPerform`, start to finish — happens inside this single
        // `withUnsafeMutableBufferPointer` closure. The pointer never escapes it,
        // so there is exactly one place that ever needs to reason about its
        // lifetime, and it does not outlive the call as the old code's separately
        // `allocate`d pointer did.
        results.withUnsafeMutableBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }

            // FIX (round 3) — WHY THE POINTER IS WRAPPED IN A BOX INSTEAD OF
            // CAPTURED DIRECTLY:
            // `UnsafeMutablePointer` does not conform to `Sendable` — not
            // because pointers are inherently unsafe to share across threads,
            // but because the compiler has no general way to know whether
            // concurrent access to the POINTEE is actually coordinated. Passed
            // straight into `concurrentPerform`'s `@Sendable` closure, that
            // produces exactly the warning this replaces:
            // "capture of 'base' with non-Sendable type ... in a '@Sendable'
            // closure". That warning is currently non-fatal only because this
            // package hasn't opted into the Swift 6 language mode; under
            // Swift 6 the identical situation is a hard error, so leaving it
            // as a bare capture would be a ticking time bomb for the next
            // toolchain upgrade.
            //
            // Wrapping the pointer in `SendableBox` (declared as
            // `@unchecked Sendable`) moves the "I have verified this is
            // actually safe" claim to exactly one place — the box's
            // declaration — instead of asking the compiler to trust an
            // un-annotated capture. The safety argument itself doesn't change:
            // `concurrentPerform` blocks until every iteration has completed,
            // and each iteration only ever writes to its own disjoint index
            // `i`, so there is no genuine race — only one the type system
            // cannot see through a raw pointer. This is the standard pattern
            // for exactly this "parallel fan-out into a preallocated buffer"
            // shape, and it compiles warning-free under both Swift 5 and
            // Swift 6 language modes.
            let box = SendableBox(pointer: baseAddress)

            DispatchQueue.concurrentPerform(iterations: count) { i in
                queries[i].withUnsafeBufferPointer { buf in
                    guard let ptr = buf.baseAddress else { return }
                    box.pointer[i] = engine.search(query: ptr, k: k, ef: ef)
                }
            }
        }

        return results
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
            throw VectorDatabaseError.notFound(id)
        }

        // 1. Update in-memory IDMap
        try idMap.updateMetadata(for: id, metadata: metadata)

        // 2. Append to WAL
        try persistenceManager?.wal?.append(
            record: WALRecord(
                opcode: .updateMetadata,
                internalID: internalID,
                timestamp: UInt64(Date().timeIntervalSince1970 * 1000),
                metadata: metadata
            ))

        try checkAndFlushWAL()
    }

    /// Retrieves a single vector and its metadata by ID.
    /// - Parameter id: The external string ID of the vector.
    /// - Returns: A tuple containing the copied vector and metadata, or `nil` if
    ///   not found, tombstoned, or the database is closed.
    public func get(id: String) -> (vector: [Float], metadata: [String: String]?)? {
        // Fix S2: the previous version wrote `try? checkNotClosed()` and discarded
        // the result, which did literally nothing. `listIDs` already guarded
        // correctly; this now matches it.
        guard !isClosed else { return nil }
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
        return try rebuildIfNeeded()
    }

    /// Rebuilds the index if the tombstone ratio warrants it.
    ///
    /// FIX K2 — THIS IS DELIBERATELY SYNCHRONOUS NOW.
    /// It used to be:
    ///
    ///     let snapshots = engine.collectLiveSnapshots()
    ///     Task.detached(priority: .utility) { try? capturedEngine.rebuild(with: snapshots) }
    ///
    /// which returned immediately and rebuilt in the background. But `rebuild`
    /// finishes by SWAPPING IN a router built purely from `snapshots`. Any
    /// `insert` the actor accepted between the snapshot and the swap landed in
    /// the old router and was thrown away with it — the record still existed in
    /// `idMap` (so `stats().liveCount` counted it) and in the WAL, but `search`
    /// could not find it and `get()` returned nil. Silent, permanent divergence
    /// between the ID map and the index.
    ///
    /// Because this actor never suspends between `collectLiveSnapshots()` and
    /// `rebuild(...)`, no mutation can interleave and nothing can be lost. The
    /// price is that the `delete`/`update` that crosses the threshold pays an
    /// O(live count) pause instead of returning instantly. That is the correct
    /// trade: a predictable, occasional pause beats losing writes. Callers who
    /// want to control exactly when that pause happens can keep the ratio low by
    /// calling `compact()` at a moment of their choosing.
    @discardableResult
    private func rebuildIfNeeded() throws -> Bool {
        guard engine.shouldRebuild else { return false }
        try engine.rebuild(with: engine.collectLiveSnapshots())
        return true
    }

    // MARK: - Batch Operations

    /// Transactionally inserts a batch of vectors.
    /// Acquires the actor lock once for the entire batch.
    /// Policy: This implements a strict all-or-nothing rollback. On any failure,
    /// any items already inserted in this batch are rolled back (from `idMap`, `Engine`, and WAL)
    /// exactly as if the batch had never been called.
    /// - Throws: If any single item fails validation (e.g. `.duplicateID`).
    public func batchInsert(_ items: [(id: String, vector: [Float], metadata: [String: String]?)])
        throws
    {
        try checkNotClosed()
        for item in items {
            try validate(id: item.id, vector: item.vector)
        }

        var insertedIDs: [String] = []
        insertedIDs.reserveCapacity(items.count)

        var throwToCaller: Error? = nil
        do {
            // Single pass across the entire batch, one fsync at the end.
            for item in items {
                try _insert(
                    id: item.id, vector: item.vector, metadata: item.metadata, flushWAL: false)
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

    /// Writes a snapshot to disk and reclaims the part of the WAL it supersedes.
    ///
    /// The heavy vector/graph copy happens synchronously in `beginSave` — still on
    /// the actor — but writes DIRECTLY into a freshly `mmap`'d destination file
    /// rather than into an intermediate heap buffer, so only the live buffers
    /// themselves are ever fully resident in heap memory. The slow part (JSON
    /// tail, checksum, fsync, atomic rename) is handed to a detached task so the
    /// actor stays responsive.
    ///
    /// FIX K1 — WAL TRUNCATION HAPPENS *AFTER* THE AWAIT, BACK ON THE ACTOR,
    /// AND ONLY UP TO THE OFFSET CAPTURED WITH THE SNAPSHOT.
    /// Previously the detached task itself called `wal.truncate()`, wiping the
    /// whole log. Two things were wrong with that: (1) records appended while the
    /// detached task was running were in the WAL but not in the snapshot, so
    /// wiping the log destroyed them outright, and (2) truncating from a
    /// background thread races with the actor's own `append` calls on the same
    /// `FileHandle`. Doing the truncation here, on the actor, with a watermark,
    /// fixes both.
    public func save() async throws {
        try checkNotClosed()
        guard let pm = persistenceManager else { return }  // No-op for in-memory mode

        let pending = try pm.beginSave(engine: engine)
        let idMapCopy = idMap

        try await Task.detached(priority: .background) {
            try pm.finishSave(pending, idMap: idMapCopy)
        }.value

        // Back on the actor. Anything appended during the await lives after
        // `walTruncationOffset` and is preserved.
        try pm.truncateWALAfterSave(upTo: pending.walTruncationOffset)
        writesSinceLastSync = 0
    }

    /// Safely shuts down the database.
    /// - Performs a best-effort `save()`.
    /// - Unmaps and closes underlying file descriptors by releasing the index.
    /// - Idempotent: safe to call multiple times.
    public func close() async {
        if isClosed { return }

        // Force a final fsync before save
        try? persistenceManager?.wal?.fsync()
        writesSinceLastSync = 0

        // Best-effort flush
        try? await save()

        isClosed = true
        // Allow the OS to reclaim file descriptors and mapped memory, since
        // VectorStorage and GraphStorage rely on MappedFile lifecycle.
        engine.close()
    }

    // MARK: - Introspection

    /// Returns current statistics.
    public func stats() -> Stats {
        // Fix S3: `onDiskSizeBytes` used to be hardcoded to `nil` with a
        // "could use FileManager" comment — a documented public field that never
        // returned a value. It now reports the combined `.vdb` + `.wal` size.
        return Stats(
            liveCount: idMap.count,
            tombstonedCount: engine.tombstonedCount,
            dimension: engine.dimension,
            onDiskSizeBytes: persistenceManager.flatMap { $0.onDiskSizeBytes() }
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
        guard
            let neighborInternalIDs = engine.inspectNeighbors(
                internalID: internalID, atLayer: layer)
        else {
            return nil
        }
        return neighborInternalIDs.compactMap { idMap.externalID(for: $0) }
    }

    internal func _inspectFullTopologySnapshot() -> (
        entryPoint: String?, entryPointLevel: Int,
        nodes: [String: (level: Int, neighborsByLayer: [[String]])]
    ) {
        let (epID, epLevel, rawNodes) = engine.inspectFullTopology()
        let mappedEP = epID.flatMap { idMap.externalID(for: $0) }

        var mappedNodes: [String: (level: Int, neighborsByLayer: [[String]])] = [:]
        for (internalID, nodeInfo) in rawNodes {
            guard let extID = idMap.externalID(for: internalID) else { continue }
            var mappedNeighbors: [[String]] = []
            for layer in nodeInfo.neighborsByLayer {
                mappedNeighbors.append(layer.compactMap { idMap.externalID(for: $0) })
            }
            mappedNodes[extID] = (level: nodeInfo.level, neighborsByLayer: mappedNeighbors)
        }
        return (entryPoint: mappedEP, entryPointLevel: epLevel, nodes: mappedNodes)
    }

    internal func setTestHookOnAfterRename(_ hook: (@Sendable () -> Void)?) {
        persistenceManager?.onAfterRename = hook
    }
}

/// A minimal wrapper that lets a raw pointer cross into a `@Sendable` closure.
///
/// See the usage site in `VectorDatabase.runParallelSearches` for the full safety
/// argument. In short: `UnsafeMutablePointer` never conforms to `Sendable`
/// because the compiler cannot verify how its pointee is accessed, so passing
/// one into an API like `DispatchQueue.concurrentPerform` — which requires a
/// `@Sendable` closure — needs an explicit, localized opt-out. `@unchecked
/// Sendable` on this one-property box is that opt-out: it makes the CLOSURE'S
/// capture trivially well-typed (the box itself is Sendable), while keeping
/// the actual "why this is safe" reasoning next to the one place that needs
/// it, rather than sprinkling `nonisolated(unsafe)` at every call site.
private struct SendableBox<T>: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<T>
}
