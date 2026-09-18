//
//  MappedSnapshot.swift
//  SwiftVectorDatabase
//

import Foundation

/// MappedSnapshot.swift — the value types used by the (single) snapshot-writing
/// path: `Engine.writeSnapshot` → `IndexRouter.writeSnapshot` →
/// `PersistenceManager.beginSave` / `finishSave`.
///
/// WHY THE SNAPSHOT IS WRITTEN STRAIGHT INTO AN `mmap`'d FILE:
/// An earlier design copied the ENTIRE live vector/graph data into brand-new
/// heap `Data` buffers (`IndexSnapshot`) so the disk-writing work could be
/// handed off to a background `Task` without blocking the actor. While that
/// `Data` existed, the live buffers AND the heap copy were both resident at
/// once — the ~2x peak RSS this library used to document as a known limitation.
///
/// The current path keeps the same "detach the slow disk I/O" property but
/// changes WHERE the copy's destination memory lives: instead of a new heap
/// allocation, the copy is written directly into an already-`mmap`'d
/// destination file (disk-backed pages, reclaimable by the OS under memory
/// pressure). The copy itself still happens synchronously under Engine's read
/// lock — exactly as expensive as the old `Data` copy, not slower — only the
/// destination changed.
///
/// FIX S1: the heap-copy path (`IndexSnapshot`, `IndexRouter.createSnapshot()`,
/// `Engine.createSnapshot()`, `PersistenceManager.save(snapshot:idMap:)`) has
/// been deleted. It had no production callers left, yet it carried a second,
/// independent implementation of the on-disk layout and of the file checksum —
/// two sources of truth for one binary format is a correctness trap, and the
/// tests that exercised it were validating code the app never ran.

/// Cheap, `Sendable` byte-size information for the three binary sections a snapshot
/// needs to write. Computing this requires no copying — it is plain arithmetic over
/// the live row counts.
struct SnapshotSectionSizes: Sendable {
    let vectorBytes: Int
    let layer0Bytes: Int
    let upperLayerByteSizes: [Int]

    var totalBinaryBytes: Int {
        vectorBytes + layer0Bytes + upperLayerByteSizes.reduce(0, +)
    }
}

/// Raw destination pointers for the three binary sections, supplied by the caller
/// (`PersistenceManager`) after it has sized and `mmap`'d a destination file based on
/// a `SnapshotSectionSizes` value. `IndexRouter.writeSnapshot` copies the live
/// vector/graph data directly into these pointers — no intermediate heap `Data`.
///
/// Lifetime note: these pointers are only valid for the duration of the
/// `writeSnapshot(makeDestination:)` call that produced them (i.e. while Engine's
/// read lock is held). They must not be retained or used afterward.
struct SnapshotDestination {
    let vector: UnsafeMutableRawPointer
    let layer0: UnsafeMutableRawPointer
    let upperLayers: [UnsafeMutableRawPointer]
}

/// Everything `PersistenceManager` needs to finish writing a snapshot's file header
/// and JSON tail, EXCEPT the (already-copied-to-destination) binary sections.
struct MappedSnapshotMetadata: @unchecked Sendable {
    let nodes: [Int32: HNSWNode]
    let entryPoint: Int32?
    let entryPointLevel: Int
    let neighborCounts: [[Int]]

    let hnswSeed: UInt64
    let hnswEfConstruction: Int
    let hnswEfSearch: Int
    let walFormatVersion: UInt32
    let metric: DistanceMetric

    let vectorCount: Int
    let capacity: Int
    let mMax0: Int
    let m: Int
    let L: Int
}
