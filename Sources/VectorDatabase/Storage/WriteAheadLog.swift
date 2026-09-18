import Darwin
import Foundation

/// WriteAheadLog.swift — Append-only durability log (insert/delete records + fsync).
///
/// ON-DISK SHAPE
///   [5-byte header: 'V' 'W' 'A' 'L' <version>]
///   then, repeated:
///     opcode        : UInt8
///     internalID    : Int32   (little-endian)
///     timestamp     : UInt64  (little-endian)
///     externalIDLen : Int32   (inserts only)
///     externalID    : UTF-8 bytes
///     vector        : dimension * Float32
///     metadataLen   : Int32   (updateMetadata / insertWithMetadata only)
///     metadata      : JSON bytes
///     checksum      : UInt64  (format version >= 2 only)
///
/// Integers are written with `withUnsafeBytes(of:)`, i.e. host byte order, and
/// read back explicitly as little-endian. Every platform this package supports
/// (iOS / macOS, arm64 and x86_64) is little-endian, so the two agree; being
/// explicit on the read side documents the format instead of leaving it to the
/// compiler's memory layout.

public enum WALOpcode: UInt8 {
    case insert = 0
    case delete = 1
    case updateMetadata = 2
    case insertWithMetadata = 3
}

public struct WALRecord {
    public let opcode: WALOpcode
    public let internalID: Int32
    public let externalID: String?
    public let timestamp: UInt64
    /// Only present for inserts. Length matches the dimension of the index.
    public let vectorData: [Float]?
    /// Only present for updateMetadata / insertWithMetadata.
    public let metadata: [String: String]?

    public init(
        opcode: WALOpcode, internalID: Int32, externalID: String? = nil, timestamp: UInt64,
        vectorData: [Float]? = nil, metadata: [String: String]? = nil
    ) {
        self.opcode = opcode
        self.internalID = internalID
        self.externalID = externalID
        self.timestamp = timestamp
        self.vectorData = vectorData
        self.metadata = metadata
    }
}

public final class WriteAheadLog: @unchecked Sendable {

    /// Size of the file header in bytes.
    static let headerSize: Int = 5

    /// The format version this build writes.
    static let currentFormatVersion: UInt8 = 2

    private static func headerBytes(version: UInt8) -> Data {
        Data([86, 87, 65, 76, version])  // 'V' 'W' 'A' 'L' <version>
    }

    private var fileHandle: FileHandle?
    private let path: URL
    private let dimension: Int
    public private(set) var version: UInt8 = 1

    public init(path: URL, dimension: Int) throws {
        self.path = path
        self.dimension = dimension

        let fm = FileManager.default
        let fileExists = fm.fileExists(atPath: path.path)
        let fileSize = (try? fm.attributesOfItem(atPath: path.path)[.size] as? UInt64) ?? 0

        if !fileExists || fileSize == 0 {
            if !fileExists {
                fm.createFile(atPath: path.path, contents: nil, attributes: nil)
            }
            let fh = try FileHandle(forUpdating: path)
            self.fileHandle = fh
            try fh.write(contentsOf: Self.headerBytes(version: Self.currentFormatVersion))
            self.version = Self.currentFormatVersion
        } else {
            let fh = try FileHandle(forUpdating: path)
            self.fileHandle = fh
            let headerData = try fh.read(upToCount: Self.headerSize)
            if let data = headerData, data.count == Self.headerSize,
                data[0] == 86, data[1] == 87, data[2] == 65, data[3] == 76
            {
                self.version = data[4]
            } else {
                self.version = 0  // Legacy, headerless
            }
            try fh.seekToEnd()
        }
    }

    deinit {
        try? fileHandle?.close()
    }

    // MARK: - Append

    public func append(record: WALRecord) throws {
        guard let fh = fileHandle else { throw VectorDatabaseError.ioError(errno: EBADF) }

        if (record.opcode == .insert || record.opcode == .insertWithMetadata)
            && record.externalID == nil
        {
            throw VectorDatabaseError.invalidParameters(
                reason: "externalID is required for insert WAL records")
        }

        let vecCount = record.vectorData?.count ?? 0
        var data = Data(capacity: 1 + 4 + 8 + 4 + 64 + vecCount * MemoryLayout<Float>.size)

        var op = record.opcode.rawValue
        data.append(&op, count: 1)

        var id = record.internalID
        data.append(withUnsafeBytes(of: &id) { Data($0) })

        var ts = record.timestamp
        data.append(withUnsafeBytes(of: &ts) { Data($0) })

        if record.opcode == .insert || record.opcode == .insertWithMetadata {
            let extData = record.externalID!.data(using: .utf8)!
            var len = Int32(extData.count)
            data.append(withUnsafeBytes(of: &len) { Data($0) })
            data.append(extData)
        }

        if record.opcode == .insert || record.opcode == .insertWithMetadata,
            let vData = record.vectorData
        {
            vData.withUnsafeBufferPointer { buf in
                data.append(buf)
            }
        }

        if record.opcode == .updateMetadata || record.opcode == .insertWithMetadata {
            if let meta = record.metadata, let encoded = try? JSONEncoder().encode(meta) {
                var len = Int32(encoded.count)
                data.append(withUnsafeBytes(of: &len) { Data($0) })
                data.append(encoded)
            } else {
                var len: Int32 = 0
                data.append(withUnsafeBytes(of: &len) { Data($0) })
            }
        }

        // Per-record FNV-1a checksum, format version >= 2 only. This is the same
        // algorithm the main snapshot file uses — applied here so a single bit-flip
        // WITHIN an otherwise well-formed record (e.g. a corrupted float in
        // vectorData, without the length fields themselves being touched) is caught
        // on replay instead of silently accepted. A record that is truncated
        // mid-write is still caught by the length guards in readAll() regardless.
        // Gated on `version` because an existing v1 WAL, opened by this code before
        // its next truncation, must keep appending v1-shaped records — mixing
        // checksum and non-checksum records under one file-level version byte would
        // make them impossible to tell apart on replay.
        if version >= 2 {
            var checksum = data.withUnsafeBytes { Checksum.fnv1a($0) }
            data.append(withUnsafeBytes(of: &checksum) { Data($0) })
        }

        do {
            try fh.write(contentsOf: data)
        } catch {
            throw VectorDatabaseError.ioError(errno: EIO)
        }
    }

    public private(set) var fsyncCallCount: Int = 0

    public func fsync() throws {
        guard let fh = fileHandle else { throw VectorDatabaseError.ioError(errno: EBADF) }
        do {
            try fh.synchronize()
            fsyncCallCount += 1
        } catch {
            throw VectorDatabaseError.ioError(errno: EIO)
        }
    }

    /// Byte offset one past the last committed record — i.e. the current end of
    /// the file.
    ///
    /// FIX K1: `PersistenceManager.beginSave` captures this at the same instant
    /// it snapshots the index, so that the matching truncation later can tell
    /// "bytes the snapshot already covers" apart from "bytes appended while the
    /// snapshot was being written to disk".
    public func endOffset() throws -> UInt64 {
        guard let fh = fileHandle else { throw VectorDatabaseError.ioError(errno: EBADF) }
        do {
            return try fh.seekToEnd()
        } catch {
            throw VectorDatabaseError.ioError(errno: EIO)
        }
    }

    // MARK: - Truncation

    /// Discards the WAL prefix that a freshly-written snapshot has superseded,
    /// while PRESERVING every byte at or after `offset`.
    ///
    /// FIX K1 — THE DURABILITY HOLE THIS CLOSES:
    /// `save()` used to unconditionally wipe the whole WAL once the snapshot file
    /// was renamed into place. But `VectorDatabase.save()` is `async`: the actor is free
    /// to accept new `insert`/`delete` calls while the snapshot is being written.
    /// Those records went into the WAL but NOT into the already-taken snapshot —
    /// and were then erased by the truncation. They existed in neither file. A
    /// crash at that point lost them permanently. (The existing
    /// `testSaveDoesNotBlockConcurrentInserts` did not catch this because it
    /// called `close()`, which performs a SECOND save that happened to rescue the
    /// record.)
    ///
    /// Keeping the tail makes the operation exactly "reclaim what is now
    /// redundant", which is all a WAL truncation should ever mean.
    public func truncate(keepingBytesAfter offset: UInt64) throws {
        guard let fh = fileHandle else { throw VectorDatabaseError.ioError(errno: EBADF) }

        do {
            let cut = max(offset, UInt64(Self.headerSize))
            let end = try fh.seekToEnd()

            var tail = Data()
            if end > cut {
                try fh.seek(toOffset: cut)
                tail = try fh.readToEnd() ?? Data()
            }

            try fh.truncate(atOffset: 0)
            try fh.seek(toOffset: 0)

            if tail.isEmpty || version == 0 {
                // Nothing to carry over (or a legacy headerless file we cannot
                // splice): start clean at the current format version.
                try fh.write(contentsOf: Self.headerBytes(version: Self.currentFormatVersion))
                self.version = Self.currentFormatVersion
            } else {
                // The carried-over records were written under the CURRENT version,
                // so the header byte must keep declaring that version — otherwise
                // replay would look for per-record checksums that aren't there (or
                // miss ones that are).
                try fh.write(contentsOf: Self.headerBytes(version: self.version))
                try fh.write(contentsOf: tail)
            }

            try fh.synchronize()
        } catch let error as VectorDatabaseError {
            throw error
        } catch {
            throw VectorDatabaseError.ioError(errno: EIO)
        }
    }

    /// Truncates the WAL to 0 bytes and rewrites the header, discarding everything.
    public func truncate() throws {
        try truncate(keepingBytesAfter: UInt64.max)
    }

    // MARK: - Replay

    /// Replays every complete, checksum-valid record in the log.
    ///
    /// FIX O3 — WHY THIS NO LONGER SLURPS THE WHOLE FILE:
    /// The previous implementation called `readToEnd()` and then carved records
    /// out of that one giant `Data` with `subdata(in:)`. With a large
    /// `walFlushInterval`, or simply a long stretch between `save()` calls, the
    /// WAL can reach hundreds of megabytes — and recovery would materialise all
    /// of it in RAM at the exact moment the app is starting up, plus a fresh copy
    /// for every field of every record.
    ///
    /// The log is now read through a 64 KiB sliding window that is compacted at
    /// each record boundary, so peak buffer usage is bounded by
    /// `chunk size + one record`, independent of total WAL size.
    public func readAll() throws -> [WALRecord] {
        guard let fh = fileHandle else { throw VectorDatabaseError.ioError(errno: EBADF) }

        let resumeOffset = try fh.offset()
        defer { try? fh.seek(toOffset: resumeOffset) }

        try fh.seek(toOffset: 0)
        var reader = ChunkedByteReader(handle: fh)

        if version > 0 {
            guard try reader.skip(Self.headerSize) else { return [] }
        }

        var records: [WALRecord] = []
        let vectorByteCount = dimension * MemoryLayout<Float>.size

        replayLoop: while true {
            reader.compactToCursor()
            reader.markRecordStart()

            guard let opcodeBytes = try reader.take(1) else { break }
            guard let opcode = WALOpcode(rawValue: opcodeBytes[0]) else {
                print(
                    "WARNING: Invalid WAL opcode \(opcodeBytes[0]), assuming mid-write crash and stopping recovery."
                )
                break
            }

            guard let idBytes = try reader.take(4) else {
                print("WARNING: Truncated WAL record (ID), assuming mid-write crash.")
                break
            }
            let internalID = Int32(bitPattern: Self.uint32LE(idBytes))

            guard let tsBytes = try reader.take(8) else {
                print("WARNING: Truncated WAL record (Timestamp), assuming mid-write crash.")
                break
            }
            let timestamp = Self.uint64LE(tsBytes)

            var externalID: String? = nil
            var vector: [Float]? = nil
            var metadata: [String: String]? = nil

            if opcode == .insert || opcode == .insertWithMetadata {
                if version == 0 {
                    throw VectorDatabaseError.legacyWALFormatNotSupported(
                        reason:
                            "Legacy WAL format (v0) is no longer supported for inserts due to missing externalID. Please delete the .wal file to proceed."
                    )
                }

                guard let lenBytes = try reader.take(4) else {
                    print(
                        "WARNING: Truncated WAL record (ExternalID Len), assuming mid-write crash.")
                    break
                }
                let length = Int(Int32(bitPattern: Self.uint32LE(lenBytes)))
                guard length >= 0 else {
                    print("WARNING: Negative WAL externalID length, stopping recovery.")
                    break
                }

                if length > 0 {
                    guard let extBytes = try reader.take(length) else {
                        print(
                            "WARNING: Truncated WAL record (ExternalID), assuming mid-write crash.")
                        break
                    }
                    externalID = String(bytes: extBytes, encoding: .utf8)
                }

                guard let vectorBytes = try reader.take(vectorByteCount) else {
                    print("WARNING: Truncated WAL record (Vector), assuming mid-write crash.")
                    break
                }
                vector = Self.floats(from: vectorBytes, count: dimension)
            }

            if opcode == .updateMetadata || opcode == .insertWithMetadata {
                guard let lenBytes = try reader.take(4) else {
                    print("WARNING: Truncated WAL record (Metadata Len), assuming mid-write crash.")
                    break
                }
                let length = Int(Int32(bitPattern: Self.uint32LE(lenBytes)))
                guard length >= 0 else {
                    print("WARNING: Negative WAL metadata length, stopping recovery.")
                    break
                }

                if length > 0 {
                    guard let metaBytes = try reader.take(length) else {
                        print("WARNING: Truncated WAL record (Metadata), assuming mid-write crash.")
                        break
                    }
                    metadata = try? JSONDecoder().decode(
                        [String: String].self, from: Data(metaBytes))
                }
            }

            // Verify the per-record checksum written by append() (format version >= 2
            // only). A mismatch or a truncated checksum both mean this record is not
            // trustworthy; stop replay here rather than accept possibly corrupted
            // data, matching how a truncated length/opcode is handled above.
            if version >= 2 {
                let computed = reader.hashOfCurrentRecord()
                guard let checksumBytes = try reader.take(8) else {
                    print("WARNING: Truncated WAL record (checksum), assuming mid-write crash.")
                    break replayLoop
                }
                let stored = Self.uint64LE(checksumBytes)
                guard stored == computed else {
                    print(
                        "WARNING: WAL record checksum mismatch (stored \(stored), computed \(computed)); stopping recovery to avoid replaying corrupted data."
                    )
                    break replayLoop
                }
            }

            records.append(
                WALRecord(
                    opcode: opcode, internalID: internalID, externalID: externalID,
                    timestamp: timestamp, vectorData: vector, metadata: metadata))
        }

        return records
    }

    // MARK: - Little-endian decoding helpers

    private static func uint32LE(_ b: ArraySlice<UInt8>) -> UInt32 {
        let base = b.startIndex
        return UInt32(b[base])
            | (UInt32(b[base + 1]) << 8)
            | (UInt32(b[base + 2]) << 16)
            | (UInt32(b[base + 3]) << 24)
    }

    private static func uint64LE(_ b: ArraySlice<UInt8>) -> UInt64 {
        let base = b.startIndex
        var value: UInt64 = 0
        for i in 0..<8 {
            value |= UInt64(b[base + i]) << (8 * UInt64(i))
        }
        return value
    }

    private static func floats(from bytes: ArraySlice<UInt8>, count: Int) -> [Float] {
        let contiguous = Array(bytes)
        return contiguous.withUnsafeBytes { raw in
            (0..<count).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
        }
    }
}

// MARK: - ChunkedByteReader

/// A forward-only sliding window over a `FileHandle`.
///
/// The buffer holds at most `chunkSize` bytes plus whatever the record currently
/// being parsed needs, because `compactToCursor()` is called at each record
/// boundary (and only there — dropping consumed bytes mid-record would invalidate
/// the range `hashOfCurrentRecord()` needs).
///
/// The buffer is `[UInt8]` rather than `Data` deliberately: `Array` always
/// re-indexes from 0 after `removeFirst`, whereas `Data` slices can retain their
/// original indices, which is a classic source of silent off-by-N bugs.
private struct ChunkedByteReader {
    private let handle: FileHandle
    private let chunkSize: Int

    private var buffer: [UInt8] = []
    private var cursor: Int = 0
    private var recordStart: Int = 0
    private var reachedEOF = false

    init(handle: FileHandle, chunkSize: Int = 64 * 1024) {
        self.handle = handle
        self.chunkSize = chunkSize
    }

    /// Ensures at least `n` unconsumed bytes are buffered, reading more from disk
    /// if needed. Returns false when the file ends first.
    private mutating func ensure(_ n: Int) throws -> Bool {
        while buffer.count - cursor < n && !reachedEOF {
            let next = try handle.read(upToCount: chunkSize) ?? Data()
            if next.isEmpty {
                reachedEOF = true
                break
            }
            buffer.append(contentsOf: next)
        }
        return buffer.count - cursor >= n
    }

    /// Consumes and returns the next `n` bytes, or nil if fewer remain.
    mutating func take(_ n: Int) throws -> ArraySlice<UInt8>? {
        guard n >= 0 else { return nil }
        guard try ensure(n) else { return nil }
        let slice = buffer[cursor..<(cursor + n)]
        cursor += n
        return slice
    }

    /// Consumes `n` bytes without returning them.
    mutating func skip(_ n: Int) throws -> Bool {
        guard try ensure(n) else { return false }
        cursor += n
        return true
    }

    /// Drops every already-consumed byte. Safe ONLY at a record boundary.
    mutating func compactToCursor() {
        if cursor > 0 {
            buffer.removeFirst(cursor)
            cursor = 0
        }
        recordStart = 0
    }

    mutating func markRecordStart() {
        recordStart = cursor
    }

    /// FNV-1a over the bytes consumed since `markRecordStart()`, computed in
    /// place with no intermediate copy.
    func hashOfCurrentRecord() -> UInt64 {
        buffer.withUnsafeBytes { raw in
            Checksum.fnv1a(UnsafeRawBufferPointer(rebasing: raw[recordStart..<cursor]))
        }
    }
}
