import Foundation
import os

/// ReadWriteLock.swift — pthread_rwlock_t wrapper for concurrent reads / exclusive writes.
/// Implementation: Phase 5 (concurrency & thread safety).
///
/// This lock solves the "Writer Starvation" pitfall (§9) on Darwin.
/// Darwin's `pthread_rwlock_t` is famously reader-preferring and non-fair. Under heavy
/// read load, a writer can wait forever while readers bypass it. We solve this by 
/// pairing the rwlock with an `os_unfair_lock` acting as a "write intent" gate.
/// Writers hold the gate for the duration of the write. Readers acquire and immediately 
/// release the gate before taking the read lock. This ensures new readers block at the 
/// gate if a writer is waiting, allowing the rwlock to drain and the writer to proceed.
final class ReadWriteLock {
    private var rwlock = pthread_rwlock_t()
    private var writeIntentLock = os_unfair_lock_s()

    init() {
        pthread_rwlock_init(&rwlock, nil)
    }

    deinit {
        pthread_rwlock_destroy(&rwlock)
    }

    func withRead<T>(_ body: () throws -> T) rethrows -> T {
        // 1. Pass through the write-intent gate to ensure we don't bypass a waiting writer.
        os_unfair_lock_lock(&writeIntentLock)
        os_unfair_lock_unlock(&writeIntentLock)

        // 2. Acquire shared read lock.
        pthread_rwlock_rdlock(&rwlock)
        defer { pthread_rwlock_unlock(&rwlock) }
        
        return try body()
    }

    func withWrite<T>(_ body: () throws -> T) rethrows -> T {
        // 1. Lock the write-intent gate to block any new readers from entering.
        os_unfair_lock_lock(&writeIntentLock)
        defer { os_unfair_lock_unlock(&writeIntentLock) }

        // 2. Acquire exclusive write lock (will succeed once existing readers finish).
        pthread_rwlock_wrlock(&rwlock)
        defer { pthread_rwlock_unlock(&rwlock) }
        
        return try body()
    }
}
