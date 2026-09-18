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
    
    // Version 2 fields
    let walFormatVersion: UInt32   // 72..76
    let hnswEfConstruction: UInt32 // 76..80
    let hnswEfSearch: UInt32       // 80..84
    let padding3: UInt32           // 84..88
    let hnswSeed: UInt64           // 88..96
    
    // Padding to 128 bytes (cache-line aligned and room for future expansion)
    let reserved4: UInt64          // 96..104
    let reserved5: UInt64          // 104..112
    let reserved6: UInt64          // 112..120
    let reserved7: UInt64          // 120..128
}

/// Fast, non-cryptographic hash for detecting file truncation or corruption.
enum Checksum {
    private static let fnv1aOffsetBasis: UInt64 = 0xcbf29ce484222325
    private static let fnv1aPrime: UInt64 = 0x100000001b3

    /// Computes the 64-bit FNV-1a hash of the given raw buffer in one pass.
    static func fnv1a(_ data: UnsafeRawBufferPointer) -> UInt64 {
        fnv1a(seed: fnv1aOffsetBasis, data)
    }

    /// Resumable variant: continues hashing `data` starting from a previously
    /// returned hash value (`seed`) instead of the initial FNV-1a offset basis.
    ///
    /// FIX (Fix 5, save() memory peak): FNV-1a's hash state is just a running
    /// accumulator (`hash = f(hash, byte)`, applied strictly left-to-right), so
    /// hashing chunk A then continuing with `seed: hashOfA` over chunk B is
    /// mathematically IDENTICAL to hashing the concatenation `A + B` in one pass —
    /// this is a basic, well-understood property of this style of streaming hash,
    /// not an approximation. `PersistenceManager` uses this to checksum the (large)
    /// binary section directly from its `mmap`'d destination, then continue over the
    /// (small) JSON tail `Data`, without ever needing a SECOND full-file `mmap` pass
    /// purely for checksumming (the old code's `save()` did exactly that second pass).
    static func fnv1a(seed: UInt64, _ data: UnsafeRawBufferPointer) -> UInt64 {
        var hash = seed
        for byte in data {
            hash = hash ^ UInt64(byte)
            hash = hash &* fnv1aPrime  // wrapping multiply
        }
        return hash
    }
}
