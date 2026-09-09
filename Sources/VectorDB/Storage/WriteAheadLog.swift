import Foundation
import Darwin

/// WriteAheadLog.swift — Append-only durability log (insert/delete records + fsync).
/// Implementation: Phase 6 (persistence / WAL + snapshot compaction).

public enum WALOpcode: UInt8 {
    case insert = 0
    case delete = 1
}

public struct WALRecord {
    public let opcode: WALOpcode
    public let internalID: Int32
    public let timestamp: UInt64
    /// Only present for inserts. Length matches the dimension of the index.
    public let vectorData: [Float]?
    
    public init(opcode: WALOpcode, internalID: Int32, timestamp: UInt64, vectorData: [Float]?) {
        self.opcode = opcode
        self.internalID = internalID
        self.timestamp = timestamp
        self.vectorData = vectorData
    }
}

public final class WriteAheadLog {
    private var fileHandle: FileHandle?
    private let path: URL
    private let dimension: Int
    
    public init(path: URL, dimension: Int) throws {
        self.path = path
        self.dimension = dimension
        
        let fm = FileManager.default
        if !fm.fileExists(atPath: path.path) {
            fm.createFile(atPath: path.path, contents: nil, attributes: nil)
        }
        
        do {
            self.fileHandle = try FileHandle(forUpdating: path)
            try self.fileHandle?.seekToEnd()
        } catch {
            throw VectorDBError.ioError(errno: errno)
        }
    }
    
    deinit {
        try? fileHandle?.close()
    }
    
    public func append(record: WALRecord) throws {
        guard let fh = fileHandle else { throw VectorDBError.ioError(errno: EBADF) }
        
        let vecCount = record.vectorData?.count ?? 0
        var data = Data(capacity: 1 + 4 + 8 + vecCount * MemoryLayout<Float>.size)
        
        var op = record.opcode.rawValue
        data.append(&op, count: 1)
        
        var id = record.internalID
        data.append(withUnsafeBytes(of: &id) { Data($0) })
        
        var ts = record.timestamp
        data.append(withUnsafeBytes(of: &ts) { Data($0) })
        
        if record.opcode == .insert, let vData = record.vectorData {
            vData.withUnsafeBufferPointer { buf in
                data.append(buf)
            }
        }
        
        do {
            try fh.write(contentsOf: data)
        } catch {
            throw VectorDBError.ioError(errno: EIO)
        }
    }
    
    public func fsync() throws {
        guard let fh = fileHandle else { throw VectorDBError.ioError(errno: EBADF) }
        do {
            try fh.synchronize()
        } catch {
            throw VectorDBError.ioError(errno: EIO)
        }
    }
    
    public func readAll() throws -> [WALRecord] {
        guard let fh = fileHandle else { throw VectorDBError.ioError(errno: EBADF) }
        
        let currentOffset = try fh.offset()
        try fh.seek(toOffset: 0)
        let data = try fh.readToEnd() ?? Data()
        try fh.seek(toOffset: currentOffset)
        
        var records: [WALRecord] = []
        var offset = 0
        
        while offset < data.count {
            // Read Opcode
            let opRaw = data[offset]
            guard let opcode = WALOpcode(rawValue: opRaw) else {
                print("WARNING: Invalid WAL opcode \(opRaw) at offset \(offset), assuming mid-write crash and stopping recovery.")
                break
            }
            offset += 1
            
            // Read ID
            guard offset + 4 <= data.count else {
                print("WARNING: Truncated WAL record (ID) at offset \(offset), assuming mid-write crash.")
                break
            }
            var id: Int32 = 0
            _ = withUnsafeMutableBytes(of: &id) { data.copyBytes(to: $0, from: offset..<offset+4) }
            offset += 4
            
            // Read Timestamp
            guard offset + 8 <= data.count else {
                print("WARNING: Truncated WAL record (Timestamp) at offset \(offset), assuming mid-write crash.")
                break
            }
            var ts: UInt64 = 0
            _ = withUnsafeMutableBytes(of: &ts) { data.copyBytes(to: $0, from: offset..<offset+8) }
            offset += 8
            
            var vector: [Float]? = nil
            if opcode == .insert {
                let vecBytes = dimension * MemoryLayout<Float>.size
                guard offset + vecBytes <= data.count else {
                    print("WARNING: Truncated WAL record (Vector) at offset \(offset), assuming mid-write crash.")
                    break
                }
                
                var floats = [Float](repeating: 0, count: dimension)
                floats.withUnsafeMutableBytes { dst in
                    _ = data.copyBytes(to: dst, from: offset..<offset+vecBytes)
                }
                vector = floats
                offset += vecBytes
            }
            
            records.append(WALRecord(opcode: opcode, internalID: id, timestamp: ts, vectorData: vector))
        }
        
        return records
    }
    
    /// Truncates the WAL file to 0 bytes. Called after a successful snapshot save.
    public func truncate() throws {
        guard let fh = fileHandle else { throw VectorDBError.ioError(errno: EBADF) }
        do {
            try fh.truncate(atOffset: 0)
        } catch {
            throw VectorDBError.ioError(errno: EIO)
        }
    }
}
