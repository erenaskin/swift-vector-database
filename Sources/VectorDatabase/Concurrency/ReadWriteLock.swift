import Foundation
import os

/// ReadWriteLock.swift — pthread_rwlock_t wrapper for concurrent reads / exclusive writes.
///
/// This lock solves the "Writer Starvation" pitfall (§9) on Darwin.
/// Darwin's `pthread_rwlock_t` is famously reader-preferring and non-fair. Under heavy
/// read load, a writer can wait forever while readers bypass it. We solve this by
/// pairing the rwlock with an `os_unfair_lock` acting as a "write intent" gate.
/// Writers hold the gate for the duration of the write. Readers acquire and immediately
/// release the gate before taking the read lock. This ensures new readers block at the
/// gate if a writer is waiting, allowing the rwlock to drain and the writer to proceed.
///
/// FIX Y3 — WHY THE LOCKS LIVE BEHIND `UnsafeMutablePointer` INSTEAD OF BEING
/// PLAIN STORED PROPERTIES:
/// The previous version stored `var rwlock = pthread_rwlock_t()` directly and
/// passed `&rwlock` to the C functions. In Swift, `&someProperty` establishes a
/// temporary `inout` access; the language does NOT guarantee that the address
/// handed to the callee is the property's permanent storage address (the
/// compiler is free to materialise a temporary and write it back). Both
/// `pthread_rwlock_t` and `os_unfair_lock` REQUIRE a stable address for their
/// whole lifetime — Apple documents this explicitly for `os_unfair_lock`. The
/// old code happened to work, but only by accident of the current optimiser.
///
/// Allocating the two lock structs on the heap once, in `init`, and only ever
/// passing that same pointer makes the address stability a guarantee rather than
/// a hope. (`OSAllocatedUnfairLock` would be the modern alternative, but it
/// requires iOS 16 / macOS 13 and this package targets iOS 15 / macOS 12.)
final class ReadWriteLock: @unchecked Sendable {
    private let rwlock: UnsafeMutablePointer<pthread_rwlock_t>
    private let writeIntentLock: os_unfair_lock_t

    init() {
        rwlock = UnsafeMutablePointer<pthread_rwlock_t>.allocate(capacity: 1)
        rwlock.initialize(to: pthread_rwlock_t())
        pthread_rwlock_init(rwlock, nil)

        writeIntentLock = os_unfair_lock_t.allocate(capacity: 1)
        writeIntentLock.initialize(to: os_unfair_lock_s())
    }

    deinit {
        pthread_rwlock_destroy(rwlock)
        rwlock.deinitialize(count: 1)
        rwlock.deallocate()

        writeIntentLock.deinitialize(count: 1)
        writeIntentLock.deallocate()
    }

    func withRead<T>(_ body: () throws -> T) rethrows -> T {
        // 1. Pass through the write-intent gate to ensure we don't bypass a waiting writer.
        os_unfair_lock_lock(writeIntentLock)
        os_unfair_lock_unlock(writeIntentLock)

        // 2. Acquire shared read lock.
        pthread_rwlock_rdlock(rwlock)
        defer { pthread_rwlock_unlock(rwlock) }

        return try body()
    }

    func withWrite<T>(_ body: () throws -> T) rethrows -> T {
        // 1. Lock the write-intent gate to block any new readers from entering.
        os_unfair_lock_lock(writeIntentLock)
        defer { os_unfair_lock_unlock(writeIntentLock) }

        // 2. Acquire exclusive write lock (will succeed once existing readers finish).
        pthread_rwlock_wrlock(rwlock)
        defer { pthread_rwlock_unlock(rwlock) }

        return try body()
    }
}
