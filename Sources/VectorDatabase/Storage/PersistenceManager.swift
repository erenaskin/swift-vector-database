import Foundation

/// PersistenceManager.swift — Orchestrates save/load/compaction of WAL + snapshot.

final class PersistenceManager: @unchecked Sendable {
    let databaseURL: URL
    let walURL: URL
    private(set) var wal: WriteAheadLog?

    private let dimension: Int

    /// A dedicated lock file (`<db>.lock`), separate from `.vdb`/`.wal` themselves,
    /// that save/load acquire via `flock(2)` so a second PROCESS (e.g. a Share
    /// Extension sharing an App Group container) cannot read a half-written
    /// `.vdb`/`.wal` or race with an in-progress save. See `FileLock.swift` for
    /// exactly what this does and does not guarantee.
    private let crossProcessLock: FileLock

    init(databaseURL: URL, dimension: Int) throws {
        self.databaseURL = databaseURL
        self.walURL = databaseURL.appendingPathExtension("wal")
        self.dimension = dimension
        self.crossProcessLock = try FileLock(path: databaseURL.appendingPathExtension("lock"))

        self.wal = try WriteAheadLog(path: self.walURL, dimension: dimension)
    }

    /// Hook for testing crash scenarios between the `.vdb` atomic rename and the
    /// WAL truncation.
    var onAfterRename: (@Sendable () -> Void)?

    struct SnapshotTail: Codable {
        let idMap: IDMap
        let nodes: [Int32: HNSWNode]
        let entryPoint: Int32?
        let entryPointLevel: Int
        let neighborCounts: [[Int]]
    }

    /// Carries everything `finishSave` needs across the actor → `Task.detached`
    /// boundary. Holds a live, still-open `MappedFile` (the binary sections were
    /// already written directly into it by `beginSave`, under Engine's read lock)
    /// plus the small metadata/sizing info needed to write the header and JSON
    /// tail afterward.
    struct PendingSave: @unchecked Sendable {
        let metadata: MappedSnapshotMetadata
        let sizes: SnapshotSectionSizes
        let tmpURL: URL
        let mapped: MappedFile
        /// `FileFormat.headerSize + sizes.totalBinaryBytes` — i.e. the byte offset
        /// where the JSON tail begins.
        let binarySize: Int
        /// The WAL's end offset AT THE MOMENT the snapshot was taken (fix K1).
        /// Everything before this is now redundant; everything at or after it was
        /// appended while the save was in flight and MUST survive the truncation.
        let walTruncationOffset: UInt64
    }

    // MARK: - Save

    /// Part 1/2 — runs SYNCHRONOUSLY on the caller's actor. Acquires the exclusive
    /// cross-process lock, creates + sizes + `mmap`s the `.tmp` destination file,
    /// copies the live vector/graph data directly into it (via
    /// `Engine.writeSnapshot`), and records the WAL's current end offset.
    ///
    /// The snapshot and the WAL offset are captured in the same uninterrupted
    /// turn, which is what makes the offset a meaningful "everything before this
    /// is already in the snapshot" watermark.
    ///
    /// The cross-process lock acquired here is released by `finishSave`, NOT by
    /// this function — the two together form one critical section that spans an
    /// `await` gap, which is why they can't be a single `withLock { ... }` call.
    func beginSave(engine: Engine) throws -> PendingSave {
        try crossProcessLock.lock(exclusive: true)
        do {
            let walOffset = try wal?.endOffset() ?? 0

            let tmpURL = databaseURL.appendingPathExtension("tmp")
            let fm = FileManager.default
            if fm.fileExists(atPath: tmpURL.path) {
                try fm.removeItem(at: tmpURL)
            }
            fm.createFile(atPath: tmpURL.path, contents: nil, attributes: nil)

            var mappedFileHolder: MappedFile?
            var binarySizeHolder = 0
            var sizesHolder: SnapshotSectionSizes?

            let metadata = try engine.writeSnapshot { sizes in
                sizesHolder = sizes
                let totalBinarySize = FileFormat.headerSize + sizes.totalBinaryBytes
                let mf = try MappedFile(path: tmpURL.path, initialSize: totalBinarySize)
                mappedFileHolder = mf
                binarySizeHolder = totalBinarySize

                var offset = FileFormat.headerSize
                let vectorPtr = mf.pointer.advanced(by: offset)
                offset += sizes.vectorBytes
                let layer0Ptr = mf.pointer.advanced(by: offset)
                offset += sizes.layer0Bytes

                var upperPtrs: [UnsafeMutableRawPointer] = []
                upperPtrs.reserveCapacity(sizes.upperLayerByteSizes.count)
                for size in sizes.upperLayerByteSizes {
                    upperPtrs.append(mf.pointer.advanced(by: offset))
                    offset += size
                }
                return SnapshotDestination(
                    vector: vectorPtr, layer0: layer0Ptr, upperLayers: upperPtrs)
            }

            guard let mapped = mappedFileHolder, let sizes = sizesHolder else {
                throw VectorDatabaseError.ioError(errno: EIO)
            }

            return PendingSave(
                metadata: metadata, sizes: sizes, tmpURL: tmpURL, mapped: mapped,
                binarySize: binarySizeHolder, walTruncationOffset: walOffset)
        } catch {
            crossProcessLock.unlock()
            throw error
        }
    }

    /// Part 2/2 — runs off the actor (inside `Task.detached`, see `VectorDatabase.save()`).
    /// Finishes writing the JSON tail, computes the checksum by hashing the
    /// ALREADY-mapped binary region directly, writes the real header, and performs
    /// the atomic rename. Always releases the cross-process lock `beginSave`
    /// acquired, even on failure.
    ///
    /// NOTE (fix K1): this deliberately does NOT touch the WAL. The WAL's
    /// `FileHandle` is owned by the actor, which may be appending to it right now;
    /// truncating it from this background thread would be a data race on the file
    /// handle's own offset. WAL reclamation happens in
    /// `truncateWALAfterSave(upTo:)`, back on the actor, once this returns.
    func finishSave(_ pending: PendingSave, idMap: IDMap) throws {
        defer { crossProcessLock.unlock() }

        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
            let processInfo = ProcessInfo.processInfo
            processInfo.performExpiringActivity(withReason: "VectorDatabase.Save") { _ in
                // ignoring expired flag for this synchronous operation
            }
        #endif

        // 1. Flush the binary sections `beginSave` already memcpy'd into the mmap.
        try pending.mapped.sync()

        // 2. Append the JSON tail (IDMap + node metadata) via a regular FileHandle.
        let fh = try FileHandle(forUpdating: pending.tmpURL)
        defer { try? fh.close() }
        try fh.seek(toOffset: UInt64(pending.binarySize))

        let tail = SnapshotTail(
            idMap: idMap,
            nodes: pending.metadata.nodes,
            entryPoint: pending.metadata.entryPoint,
            entryPointLevel: pending.metadata.entryPointLevel,
            neighborCounts: pending.metadata.neighborCounts
        )
        let tailData = try JSONEncoder().encode(tail)
        try fh.write(contentsOf: tailData)
        try fh.synchronize()

        // 3. Checksum: hash the binary region directly from `pending.mapped.pointer`
        //    then CONTINUE the same running FNV-1a hash over `tailData`.
        let binaryBufferStart = pending.mapped.pointer.advanced(by: FileFormat.headerSize)
        let binaryBufferCount = pending.binarySize - FileFormat.headerSize
        let binaryBuffer = UnsafeRawBufferPointer(
            start: binaryBufferStart, count: binaryBufferCount)
        var checksum = Checksum.fnv1a(binaryBuffer)
        checksum = tailData.withUnsafeBytes { Checksum.fnv1a(seed: checksum, $0) }

        let metricByte: UInt8 = {
            switch pending.metadata.metric {
            case .dotProduct: return 0
            case .cosine: return 1
            case .euclidean: return 2
            }
        }()

        let vectorSectionOffset = UInt64(FileFormat.headerSize)
        let graphSectionOffset = vectorSectionOffset + UInt64(pending.sizes.vectorBytes)
        let idMapSectionOffset = UInt64(pending.binarySize)

        // 4. Write the real header.
        var header = FileHeader(
            magicBytes: FileFormat.magicBytes,
            formatVersion: 2,
            dimension: UInt32(dimension),
            vectorCount: UInt32(pending.metadata.vectorCount),
            capacity: UInt32(pending.metadata.capacity),
            metric: metricByte,
            padding1: 0,
            padding2: 0,
            hnswM: UInt32(pending.metadata.m),
            hnswMmax0: UInt32(pending.metadata.mMax0),
            entryPointID: pending.metadata.entryPoint ?? -1,
            entryPointLevel: UInt32(pending.metadata.entryPointLevel),
            vectorSectionOffset: vectorSectionOffset,
            graphSectionOffset: graphSectionOffset,
            idMapSectionOffset: idMapSectionOffset,
            checksum: checksum,
            walFormatVersion: 1,
            hnswEfConstruction: UInt32(pending.metadata.hnswEfConstruction),
            hnswEfSearch: UInt32(pending.metadata.hnswEfSearch),
            padding3: 0,
            hnswSeed: pending.metadata.hnswSeed,
            reserved4: 0, reserved5: 0, reserved6: 0, reserved7: 0
        )

        try fh.seek(toOffset: 0)
        let headerData = withUnsafeBytes(of: &header) { Data($0) }
        // Every downstream section offset in this file is computed from the
        // FileFormat.headerSize CONSTANT, not from this struct's actual in-memory
        // size, so a drift between the two would silently misalign the file.
        // `fatalError` rather than `precondition` because a release build may be
        // compiled with `-Ounchecked`, which strips `assert`/`precondition`.
        guard headerData.count == FileFormat.headerSize else {
            fatalError(
                "FileHeader in-memory size (\(headerData.count) bytes) does not match "
                    + "FileFormat.headerSize (\(FileFormat.headerSize) bytes). This is a "
                    + "library bug: FileHeader was changed without updating "
                    + "FileFormat.headerSize, which would silently corrupt every file "
                    + "written by this build.")
        }
        try fh.write(contentsOf: headerData)
        try fh.synchronize()

        // 5. Atomic rename.
        let fm = FileManager.default
        if fm.fileExists(atPath: databaseURL.path) {
            _ = try fm.replaceItemAt(databaseURL, withItemAt: pending.tmpURL)
        } else {
            try fm.moveItem(at: pending.tmpURL, to: databaseURL)
        }

        onAfterRename?()
    }

    /// Reclaims the WAL prefix that `pending`'s snapshot has superseded.
    ///
    /// MUST be called from the same isolation domain that appends to the WAL
    /// (i.e. the `VectorDatabase` actor), and only after `finishSave` succeeded.
    /// Records appended while the save was in flight live after `offset` and are
    /// preserved (fix K1).
    func truncateWALAfterSave(upTo offset: UInt64) throws {
        try wal?.truncate(keepingBytesAfter: offset)
        try wal?.fsync()
    }

    /// Convenience used by tests and by any caller that wants a blocking save.
    /// Runs the exact same three steps the async `VectorDatabase.save()` does, so tests
    /// exercise the production path rather than a parallel implementation.
    func saveSynchronously(engine: Engine, idMap: IDMap) throws {
        let pending = try beginSave(engine: engine)
        try finishSave(pending, idMap: idMap)
        try truncateWALAfterSave(upTo: pending.walTruncationOffset)
    }

    /// Total bytes this database currently occupies on disk (`.vdb` + `.wal`),
    /// or nil if neither file exists yet.
    func onDiskSizeBytes() -> Int? {
        let fm = FileManager.default
        var total = 0
        var found = false
        for url in [databaseURL, walURL] {
            if let attributes = try? fm.attributesOfItem(atPath: url.path),
                let size = attributes[.size] as? UInt64
            {
                total += Int(size)
                found = true
            }
        }
        return found ? total : nil
    }

    // MARK: - Load

    func load(fallbackMetric: DistanceMetric, fallbackParams: HNSWParameters) throws -> (
        HNSWIndex, IDMap
    )? {
        try crossProcessLock.withLock(exclusive: false) {
            try _load(fallbackMetric: fallbackMetric, fallbackParams: fallbackParams)
        }
    }

    private func _load(fallbackMetric: DistanceMetric, fallbackParams: HNSWParameters) throws -> (
        HNSWIndex, IDMap
    )? {
        let fm = FileManager.default
        var index: HNSWIndex
        var idMap: IDMap

        if fm.fileExists(atPath: databaseURL.path) {

            // Get File Size
            let attrs = try fm.attributesOfItem(atPath: databaseURL.path)
            guard let fileSize = attrs[.size] as? UInt64, fileSize >= FileFormat.headerSize else {
                throw VectorDatabaseError.corruptFile(reason: "File too small")
            }

            // Mmap the file
            let mapped = try MappedFile(path: databaseURL.path, initialSize: Int(fileSize))
            let ptr = mapped.pointer

            // Read header
            let header = ptr.assumingMemoryBound(to: FileHeader.self).pointee
            guard header.magicBytes == FileFormat.magicBytes else {
                throw VectorDatabaseError.corruptFile(reason: "Invalid magic bytes")
            }
            guard header.formatVersion <= 2 else {
                throw VectorDatabaseError.unsupportedFileVersion(
                    found: header.formatVersion, supported: 2)
            }

            // FIX K4 — DIMENSION MUST MATCH WHAT THE CALLER ASKED FOR.
            // `self.dimension` is the value the caller passed to `VectorDatabase.init`,
            // and it is ALSO the value this manager handed to `WriteAheadLog`,
            // which uses it to decide how many bytes each WAL record's vector
            // occupies. If it disagrees with the file's own dimension, WAL replay
            // parses records at the wrong byte boundaries and silently produces
            // garbage. Previously nothing checked this at all: the index was
            // rebuilt at `header.dimension` while the WAL kept using the caller's.
            guard Int(header.dimension) == dimension else {
                throw VectorDatabaseError.dimensionMismatch(
                    expected: Int(header.dimension), got: dimension)
            }

            // Verify checksum (Pitfall)
            let bodySize = Int(fileSize) - FileFormat.headerSize
            let bodyPointer = ptr.advanced(by: FileFormat.headerSize)
            let bodyBuffer = UnsafeRawBufferPointer(start: bodyPointer, count: bodySize)
            let checksum = Checksum.fnv1a(bodyBuffer)
            guard checksum == header.checksum else {
                throw VectorDatabaseError.corruptFile(reason: "Checksum validation failed")
            }

            // Restore metric
            let metric: DistanceMetric = {
                switch header.metric {
                case 0: return .dotProduct
                case 1: return .cosine
                case 2: return .euclidean
                default: return .euclidean  // safe fallback
                }
            }()

            // Read Tail (JSON)
            let tailData = Data(
                bytesNoCopy: ptr.advanced(by: Int(header.idMapSectionOffset)),
                count: Int(fileSize - header.idMapSectionOffset), deallocator: .none)
            let tail = try JSONDecoder().decode(SnapshotTail.self, from: tailData)

            let seed = header.formatVersion >= 2 ? header.hnswSeed : fallbackParams.seed
            let efC =
                header.formatVersion >= 2
                ? Int(header.hnswEfConstruction) : fallbackParams.efConstruction
            let efS = header.formatVersion >= 2 ? Int(header.hnswEfSearch) : fallbackParams.efSearch

            index = HNSWIndex(
                dimension: Int(header.dimension), metric: metric,
                params: HNSWParameters(
                    M: Int(header.hnswM), efConstruction: efC, efSearch: efS, seed: seed))

            // Hook up Mapped VectorStorage
            index.vectorStorage = VectorStorage(
                dimension: Int(header.dimension),
                capacity: Int(header.vectorCount),
                mappedFile: mapped,
                offset: Int(header.vectorSectionOffset),
                count: Int(header.vectorCount)
            )

            // Hook up Mapped GraphStorage.
            //
            // `capacity` must be `header.vectorCount` (NOT `header.capacity`): the
            // graph section is persisted at exactly `graphStorage.count` rows, so
            // that is the mapped region's true on-disk size. Telling GraphStorage
            // the truth here is what makes the next `ensureUpperLayerAllocated()`
            // or `grow()` past that point fall back to a private heap allocation
            // instead of reading/writing past the end of the mapped file into the
            // JSON tail bytes.
            index.graphStorage = GraphStorage(
                m: Int(header.hnswM),
                mMax0: Int(header.hnswMmax0),
                capacity: Int(header.vectorCount),
                count: Int(header.vectorCount),
                mappedFile: mapped,
                offset: Int(header.graphSectionOffset),
                neighborCounts: tail.neighborCounts
            )

            index.loadNodes(from: tail.nodes)
            index.entryPoint = tail.entryPoint == -1 ? nil : tail.entryPoint
            index.entryPointLevel = tail.entryPointLevel

            idMap = tail.idMap

        } else {
            guard fm.fileExists(atPath: walURL.path),
                let attrs = try? fm.attributesOfItem(atPath: walURL.path),
                let size = attrs[.size] as? UInt64, size > 5
            else {
                return nil
            }

            index = HNSWIndex(dimension: dimension, metric: fallbackMetric, params: fallbackParams)
            idMap = IDMap()
        }

        try replayWAL(into: &index, idMap: &idMap)
        return (index, idMap)
    }

    /// Replays the WAL on top of an already-loaded (or freshly created) index.
    private func replayWAL(into index: inout HNSWIndex, idMap: inout IDMap) throws {
        guard let records = try wal?.readAll(), !records.isEmpty else { return }

        // FIX K3 — NORMALIZE ACCORDING TO THE INDEX'S OWN METRIC.
        // This used to test `fallbackMetric`, i.e. whatever the caller happened to
        // pass to `VectorDatabase.init` — whose DEFAULT is `.cosine`. So opening an
        // existing `.euclidean` store with `VectorDatabase(dimension:path:)` (no explicit
        // metric) silently L2-normalized every WAL-replayed vector, corrupting them
        // relative to the ones already in the snapshot. The authoritative metric is
        // the one recorded in the file header, which is exactly what `index.metric`
        // is by this point.
        let metric = index.metric

        for record in records {
            switch record.opcode {
            case .insert, .insertWithMetadata:
                guard let extID = record.externalID else { continue }

                if idMap.internalID(for: extID) != nil {
                    continue  // Already present (snapshot covered it); skip double replay.
                }

                idMap.restoreMapping(
                    externalID: extID, internalID: record.internalID, metadata: record.metadata)

                if let vData = record.vectorData {
                    var finalData = vData
                    if metric == .cosine {
                        finalData.withUnsafeMutableBufferPointer { buf in
                            if let ptr = buf.baseAddress {
                                VectorMath.normalize(ptr, buf.count)
                            }
                        }
                    }
                    finalData.withUnsafeBufferPointer { buf in
                        guard let baseAddress = buf.baseAddress else { return }
                        try? index.insert(internalID: record.internalID, vector: baseAddress)
                    }
                }

            case .delete:
                try? index.remove(internalID: record.internalID)
                if let externalID = idMap.externalID(for: record.internalID) {
                    try? idMap.remove(externalID: externalID)
                }

            case .updateMetadata:
                if let externalID = idMap.externalID(for: record.internalID) {
                    try? idMap.updateMetadata(for: externalID, metadata: record.metadata)
                }
            }
        }
    }
}
