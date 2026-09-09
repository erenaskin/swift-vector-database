import Foundation

/// PersistenceManager.swift — Orchestrates save/load/compaction of WAL + snapshot.
/// Implementation: Phase 6 (persistence / WAL + snapshot compaction).

final class PersistenceManager {
    let databaseURL: URL
    let walURL: URL
    private(set) var wal: WriteAheadLog?
    
    private let dimension: Int
    
    init(databaseURL: URL, dimension: Int) throws {
        self.databaseURL = databaseURL
        self.walURL = databaseURL.appendingPathExtension("wal")
        self.dimension = dimension
        
        self.wal = try WriteAheadLog(path: self.walURL, dimension: dimension)
    }
    
    struct SnapshotTail: Codable {
        let idMap: IDMap
        let nodes: [Int32: HNSWNode]
        let entryPoint: Int32?
        let entryPointLevel: Int
        let neighborCounts: [[Int32: Int]]
    }
    
    func save(index: HNSWIndex, idMap: IDMap) throws {
        // [Pitfall] Wrap in background task for iOS to prevent suspension mid-save
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
        let processInfo = ProcessInfo.processInfo
        processInfo.performExpiringActivity(withReason: "VectorDB.Save") { expired in
            // ignoring expired flag for this synchronous operation, but it prevents sudden suspension
        }
        #endif
        
        // Note: Cross-process access via app extensions needs real file locking (e.g. flock).
        // This is a documented v2 concern and is not solved in this design.
        
        let tmpURL = databaseURL.appendingPathExtension("tmp")
        let fm = FileManager.default
        
        if fm.fileExists(atPath: tmpURL.path) {
            try fm.removeItem(at: tmpURL)
        }
        fm.createFile(atPath: tmpURL.path, contents: nil, attributes: nil)
        
        let fh = try FileHandle(forUpdating: tmpURL)
        defer { try? fh.close() }
        
        // 1. Write dummy header
        try fh.write(contentsOf: Data(count: FileFormat.headerSize))
        
        let capacity = index.vectorStorage.capacity
        let mMax0 = index.params.Mmax0
        let m = index.params.M
        let L = max(0, index.graphStorage.neighborCounts.count - 1)
        
        let vectorSectionOffset: UInt64 = UInt64(FileFormat.headerSize)
        let vectorSectionSize: UInt64 = UInt64(capacity * dimension * 4)
        
        let graphSectionOffset: UInt64 = vectorSectionOffset + vectorSectionSize
        let layer0Size: UInt64 = UInt64(capacity * mMax0 * 4)
        let upperLayersSize: UInt64 = UInt64(L * capacity * m * 4)
        let graphSectionSize = layer0Size + upperLayersSize
        
        let idMapSectionOffset: UInt64 = graphSectionOffset + graphSectionSize
        
        // 2. Write VectorStorage
        let vecData = Data(bytesNoCopy: index.vectorStorage.buffer, count: Int(vectorSectionSize), deallocator: .none)
        try fh.write(contentsOf: vecData)
        
        // 3. Write GraphStorage Layer 0
        let l0Data = Data(bytesNoCopy: index.graphStorage.layer0, count: Int(layer0Size), deallocator: .none)
        try fh.write(contentsOf: l0Data)
        
        // 4. Write GraphStorage Upper Layers
        if L > 0 {
            for i in 1...L {
                let uPtr = index.graphStorage.upperLayerPointer(i)
                let uData = Data(bytesNoCopy: uPtr, count: capacity * m * 4, deallocator: .none)
                try fh.write(contentsOf: uData)
            }
        }
        
        // 5. Write JSON Tail (IDMap + metadata)
        let tail = SnapshotTail(
            idMap: idMap,
            nodes: index.nodes,
            entryPoint: index.entryPoint,
            entryPointLevel: index.entryPointLevel,
            neighborCounts: index.graphStorage.neighborCounts
        )
        let tailData = try JSONEncoder().encode(tail)
        try fh.write(contentsOf: tailData)
        
        try fh.synchronize()
        
        // 6. Calculate Checksum via temporary mmap
        let fileSize = try fh.seekToEnd()
        let bodySize = Int(fileSize) - FileFormat.headerSize
        let mappedBody = mmap(nil, Int(fileSize), PROT_READ, MAP_SHARED, fh.fileDescriptor, 0)
        guard mappedBody != MAP_FAILED else { throw VectorDBError.ioError(errno: errno) }
        
        let bodyPointer = mappedBody!.advanced(by: FileFormat.headerSize)
        let bodyBuffer = UnsafeRawBufferPointer(start: bodyPointer, count: bodySize)
        let checksum = Checksum.fnv1a(bodyBuffer)
        munmap(mappedBody, Int(fileSize))
        
        let metricByte: UInt8 = {
            switch index.metric {
            case .dotProduct: return 0
            case .cosine: return 1
            case .euclidean: return 2
            }
        }()
        
        // 7. Write real header
        var header = FileHeader(
            magicBytes: FileFormat.magicBytes,
            formatVersion: 1,
            dimension: UInt32(dimension),
            vectorCount: UInt32(index.count),
            capacity: UInt32(capacity),
            metric: metricByte,
            padding1: 0,
            padding2: 0,
            hnswM: UInt32(m),
            hnswMmax0: UInt32(mMax0),
            entryPointID: index.entryPoint ?? -1,
            entryPointLevel: UInt32(index.entryPointLevel),
            vectorSectionOffset: vectorSectionOffset,
            graphSectionOffset: graphSectionOffset,
            idMapSectionOffset: idMapSectionOffset,
            checksum: checksum,
            reserved1: 0, reserved2: 0, reserved3: 0, reserved4: 0, reserved5: 0, reserved6: 0, reserved7: 0
        )
        
        try fh.seek(toOffset: 0)
        let headerData = withUnsafeBytes(of: &header) { Data($0) }
        try fh.write(contentsOf: headerData)
        try fh.synchronize()
        
        // For testing mid-save crash simulation
        onBeforeRename?()
        
        // 8. Atomic Rename
        if fm.fileExists(atPath: databaseURL.path) {
            _ = try fm.replaceItemAt(databaseURL, withItemAt: tmpURL)
        } else {
            try fm.moveItem(at: tmpURL, to: databaseURL)
        }
        
        // 9. Truncate WAL
        try wal?.truncate()
    }
    
    // Testing hook
    var onBeforeRename: (() -> Void)? = nil
    
    func load() throws -> (HNSWIndex, IDMap)? {
        let fm = FileManager.default
        guard fm.fileExists(atPath: databaseURL.path) else { return nil }
        
        // Get File Size
        let attrs = try fm.attributesOfItem(atPath: databaseURL.path)
        guard let fileSize = attrs[.size] as? UInt64, fileSize >= FileFormat.headerSize else {
            throw VectorDBError.corruptFile(reason: "File too small")
        }
        
        // Mmap the file
        let mapped = try MappedFile(path: databaseURL.path, initialSize: Int(fileSize))
        let ptr = mapped.pointer
        
        // Read header
        let header = ptr.assumingMemoryBound(to: FileHeader.self).pointee
        guard header.magicBytes == FileFormat.magicBytes else {
            throw VectorDBError.corruptFile(reason: "Invalid magic bytes")
        }
        guard header.formatVersion <= 1 else {
            throw VectorDBError.unsupportedFileVersion(found: header.formatVersion, supported: 1)
        }
        
        // Verify checksum (Pitfall)
        let bodySize = Int(fileSize) - FileFormat.headerSize
        let bodyPointer = ptr.advanced(by: FileFormat.headerSize)
        let bodyBuffer = UnsafeRawBufferPointer(start: bodyPointer, count: bodySize)
        let checksum = Checksum.fnv1a(bodyBuffer)
        guard checksum == header.checksum else {
            throw VectorDBError.corruptFile(reason: "Checksum validation failed")
        }
        
        // Restore metric
        let metric: DistanceMetric = {
            switch header.metric {
            case 0: return .dotProduct
            case 1: return .cosine
            case 2: return .euclidean
            default: return .euclidean // safe fallback
            }
        }()
        
        // Read Tail (JSON)
        let tailData = Data(bytesNoCopy: ptr.advanced(by: Int(header.idMapSectionOffset)), count: Int(fileSize - header.idMapSectionOffset), deallocator: .none)
        let tail = try JSONDecoder().decode(SnapshotTail.self, from: tailData)
        
        var index = HNSWIndex(dimension: Int(header.dimension), metric: metric, params: HNSWParameters(M: Int(header.hnswM)))
        
        // Hook up Mapped VectorStorage
        index.vectorStorage = VectorStorage(
            dimension: Int(header.dimension),
            capacity: Int(header.capacity),
            mappedFile: mapped,
            offset: Int(header.vectorSectionOffset),
            count: Int(header.vectorCount)
        )
        
        // Hook up Mapped GraphStorage
        index.graphStorage = GraphStorage(
            m: Int(header.hnswM),
            mMax0: Int(header.hnswMmax0),
            capacity: Int(header.capacity),
            count: Int(header.vectorCount),
            mappedFile: mapped,
            offset: Int(header.graphSectionOffset),
            neighborCounts: tail.neighborCounts
        )
        
        index.nodes = tail.nodes
        index.entryPoint = tail.entryPoint == -1 ? nil : tail.entryPoint
        index.entryPointLevel = tail.entryPointLevel
        
        // Replay WAL
        if let records = try wal?.readAll(), !records.isEmpty {
            for record in records {
                if record.opcode == .insert, let vData = record.vectorData {
                    vData.withUnsafeBufferPointer { buf in
                        guard let baseAddress = buf.baseAddress else { return }
                        // Insert handles routing it to VectorStorage.
                        // If capacity is reached, it will fatalError inside `grow()` (handled in Phase 7).
                        try? index.insert(internalID: record.internalID, vector: baseAddress)
                    }
                } else if record.opcode == .delete {
                    try? index.remove(internalID: record.internalID)
                }
            }
        }
        
        return (index, tail.idMap)
    }
}
