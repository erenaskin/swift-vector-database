//
//  FileLock.swift
//  SwiftVectorDatabase
//
//  Created by Eren AŞKIN on 14.09.2026.
//

import Darwin
import Foundation

/// FileLock.swift — Advisory cross-process locking via BSD `flock(2)`.
///
/// WHY THIS IS NEEDED:
/// `Concurrency/ReadWriteLock.swift` protects `Engine` against data races BETWEEN
/// THREADS WITHIN ONE PROCESS. It gives zero protection if a second, independent
/// process — e.g. a Share Extension, a Today Widget, or a second launch of the same
/// App Group — opens the SAME `.vdb`/`.wal` files at the same time. `flock(2)` is a
/// kernel-level lock that works ACROSS processes, as long as every participant uses
/// it (see LIMITATION below).
///
/// SCOPE (what this does and does NOT solve):
/// This guards the two moments where a torn/partial read across processes would be
/// most damaging:
///   - `save()` takes an EXCLUSIVE lock for its entire write→checksum→atomic-rename→
///     WAL-truncate sequence, so a second process cannot start its own `save()` (or a
///     `load()`) in the middle of that sequence.
///   - `load()` takes a SHARED lock, so multiple processes can read concurrently, but
///     none can read while a writer holds the exclusive lock.
/// This does NOT provide multi-process write serialization for individual
/// `insert`/`delete`/`update` calls in between saves — concurrent multi-process
/// MUTATION of the same store is still out of scope for v1 (see README "Known
/// Limitations"). Each process's own in-memory WAL/index state is only synchronized
/// with the others at `save()`/`load()` boundaries.
///
/// LIMITATION — "advisory" means cooperative:
/// `flock` only blocks OTHER code that also calls `flock` on the same file. It does
/// NOT prevent a process that ignores locking entirely (or a non-`VectorDatabase` tool)
/// from reading/writing the file underneath a lock holder. This is a real, but
/// standard and well-understood, limitation of POSIX advisory locking.
final class FileLock: @unchecked Sendable {
    private let fd: Int32

    /// Opens (creating if necessary) a dedicated lock file NEXT TO the database,
    /// rather than locking the `.vdb`/`.wal` files directly. This keeps locking
    /// fully decoupled from the atomic-rename dance `PersistenceManager.save()`
    /// performs on `.vdb` (locking a path that gets renamed out from under you is a
    /// classic flock footgun — the new file at that path would be a DIFFERENT inode,
    /// silently unguarded by a lock taken before the rename).
    init(path: URL) throws {
        let opened = open(path.path, O_CREAT | O_RDWR, 0o666)
        guard opened != -1 else { throw VectorDatabaseError.ioError(errno: errno) }
        self.fd = opened
    }

    deinit {
        close(fd)
    }

    /// Blocks until the lock is acquired. `exclusive: true` for writers (`save()`),
    /// `false` for shared/concurrent readers (`load()`).
    ///
    /// Not `private`: `PersistenceManager`'s `beginSave`/`finishSave` split (Fix 5)
    /// needs to acquire this lock in `beginSave` (still on the actor) and release it
    /// only after `finishSave` completes (inside a detached `Task`, after an `await`
    /// gap) — a single synchronous `withLock { ... }` call cannot span that gap, so
    /// that flow calls `lock`/`unlock` directly instead, bracketing them with its own
    /// `defer`. Simpler, fully-synchronous call sites (like `load()`) should still
    /// prefer `withLock` below.
    func lock(exclusive: Bool) throws {
        let operation: Int32 = exclusive ? LOCK_EX : LOCK_SH
        while flock(fd, operation) != 0 {
            // A signal interrupting the blocking wait is not a real failure — retry.
            guard errno == EINTR else { throw VectorDatabaseError.ioError(errno: errno) }
        }
    }

    func unlock() {
        flock(fd, LOCK_UN)
    }

    /// Runs `body` while holding the lock, always releasing it afterward — including
    /// if `body` throws.
    func withLock<T>(exclusive: Bool, _ body: () throws -> T) throws -> T {
        try lock(exclusive: exclusive)
        defer { unlock() }
        return try body()
    }
}
