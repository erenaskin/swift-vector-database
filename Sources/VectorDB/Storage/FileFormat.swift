import Foundation

/// FileFormat.swift — Binary file format header layout constants & magic bytes.
/// Implementation: Phase 6 (persistence / file format).
///
/// SECTION LAYOUT (§10.1–§10.2):
/// 1. Header (128 bytes): Metadata, parameters, section offsets, and checksum.
/// 2. Vector Section: Contiguous `capacity * dimension * 4` bytes.
/// 3. Graph Section: 
///    - Layer 0: Contiguous `capacity * mMax0 * 4` bytes.
///    - Upper Layers: Sparse block `L * capacity * m * 4` bytes immediately following Layer 0.
/// 4. ID Map Section: JSON metadata blob mapping Strings to internal Int32 IDs.
///    (Explicitly NOT mmap-addressed; fully loaded into memory on startup since it is small).
///
/// CHECKSUM ALGORITHM:
/// We use the 64-bit FNV-1a hash. It is exceptionally fast, requires no external C libraries 
/// (like zlib for CRC32), and is perfectly sufficient for detecting accidental file truncation 
/// or bit-rot corruption without cryptographic overhead.

enum FileFormat {
    static let magicBytes: UInt32 = 0x56444231 // "VDB1"
    static let headerSize: Int = 128
}

/// The exact 128-byte layout of the database file header.
/// Must be tightly packed and safely reinterpret-castable from raw bytes.
struct FileHeader {
    let magicBytes: UInt32         // 0..4
    let formatVersion: UInt32      // 4..8
    let dimension: UInt32          // 8..12
    let vectorCount: UInt32        // 12..16
    
    let capacity: UInt32           // 16..20
    let metric: UInt8              // 20..21 (0=DotProduct, 1=Cosine, 2=Euclidean)
    let padding1: UInt8            // 21..22
    let padding2: UInt16           // 22..24
    
    let hnswM: UInt32              // 24..28
    let hnswMmax0: UInt32          // 28..32
    
    let entryPointID: Int32        // 32..36
    let entryPointLevel: UInt32    // 36..40
    
    let vectorSectionOffset: UInt64 // 40..48
    let graphSectionOffset: UInt64  // 48..56
    let idMapSectionOffset: UInt64  // 56..64
    
    let checksum: UInt64           // 64..72
    
    // Padding to 128 bytes (cache-line aligned and room for future expansion)
    let reserved1: UInt64          // 72..80
    let reserved2: UInt64          // 80..88
    let reserved3: UInt64          // 88..96
    let reserved4: UInt64          // 96..104
    let reserved5: UInt64          // 104..112
    let reserved6: UInt64          // 112..120
    let reserved7: UInt64          // 120..128
}

/// Fast, non-cryptographic hash for detecting file truncation or corruption.
enum Checksum {
    /// Computes the 64-bit FNV-1a hash of the given raw buffer.
    static func fnv1a(_ data: UnsafeRawBufferPointer) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x100000001b3
        
        for byte in data {
            hash = hash ^ UInt64(byte)
            hash = hash &* prime // wrapping multiply
        }
        
        return hash
    }
}
