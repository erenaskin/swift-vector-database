import Darwin
import Foundation

/// MappedFile.swift — Memory-mapped file wrapper (§10.3: `open`, `fstat`,
/// `ftruncate`, `mmap`, `msync`, `munmap`).
///
/// Marked `@unchecked Sendable` so `PersistenceManager` can hand a freshly
/// created `MappedFile` across a `Task.detached` boundary during `save()` (see
/// the `beginSave`/`finishSave` split in PersistenceManager.swift). Not
/// internally thread-safe: callers must ensure only one "owner" touches a given
/// instance at a time — the save() handoff satisfies this by construction (the
/// actor stops touching it the moment it hands the instance to the detached
/// task).
///
/// FIX S2: a `resize(to:)` method used to live here. It had no callers anywhere
/// in Sources, Tests or Benchmarks — both `VectorStorage` and `GraphStorage`
/// respond to growth by detaching from the mapping and switching to a private
/// heap allocation instead. Since nothing ever remaps, `pointer` and `size` are
/// now `let` for the lifetime of the instance, which removes the whole "the base
/// address may change after a resize" hazard class rather than merely
/// documenting it.
public final class MappedFile: @unchecked Sendable {
    private let fd: Int32
    public let size: Int
    public let pointer: UnsafeMutableRawPointer

    public init(path: String, initialSize: Int) throws {
        // All syscalls happen in a static helper that either fully succeeds or
        // throws having already closed its own descriptor, so this initializer
        // never has to throw from a half-initialized state.
        let opened = try Self.openAndMap(path: path, initialSize: initialSize)
        self.fd = opened.fd
        self.size = opened.size
        self.pointer = opened.pointer
    }

    private static func openAndMap(path: String, initialSize: Int) throws
        -> (fd: Int32, size: Int, pointer: UnsafeMutableRawPointer)
    {
        // `mmap` with a zero (or negative) length is invalid on POSIX (EINVAL) and
        // the failure mode is OS/version-dependent rather than a clean, predictable
        // error — reject it explicitly up front instead of letting `mmap` fail in a
        // confusing way.
        guard initialSize > 0 else {
            throw VectorDatabaseError.invalidParameters(
                reason: "MappedFile initialSize must be greater than 0 (got \(initialSize)).")
        }

        // 1. Open or create the file.
        // O_CLOEXEC ensures this descriptor is NOT inherited by a child process
        // created via fork/exec after this file is opened — without it, a forked
        // child would hold its own reference to the same open file description,
        // which is never what a library-internal storage handle should allow.
        let descriptor = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0o666)
        guard descriptor != -1 else {
            throw VectorDatabaseError.ioError(errno: errno)
        }

        // 2. Check current size.
        var statBuf = stat()
        guard fstat(descriptor, &statBuf) == 0 else {
            let saved = errno
            close(descriptor)
            throw VectorDatabaseError.ioError(errno: saved)
        }

        var currentSize = Int(statBuf.st_size)

        // 3. Grow the file if it is smaller than required.
        if currentSize < initialSize {
            guard ftruncate(descriptor, off_t(initialSize)) == 0 else {
                let saved = errno
                close(descriptor)
                throw VectorDatabaseError.ioError(errno: saved)
            }
            currentSize = initialSize
        }

        // 4. Memory map the file.
        let mapped = mmap(nil, currentSize, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0)
        guard mapped != MAP_FAILED, let mappedPointer = mapped else {
            let saved = errno
            close(descriptor)
            throw VectorDatabaseError.ioError(errno: saved)
        }

        return (fd: descriptor, size: currentSize, pointer: mappedPointer)
    }

    deinit {
        munmap(pointer, size)
        close(fd)
    }

    /// Flushes all dirty mapped pages to disk.
    public func sync() throws {
        guard msync(pointer, size, MS_SYNC) == 0 else {
            throw VectorDatabaseError.ioError(errno: errno)
        }
    }
}
