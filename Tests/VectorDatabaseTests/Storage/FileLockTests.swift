//
//  FileLockTests.swift
//  SwiftVectorDatabase
//
//  Created by Eren AŞKIN on 14.09.2026.
//

import XCTest

@testable import VectorDatabase

/// FileLockTests.swift — Verifies the cross-process `flock(2)` wrapper added to fix
/// the "no cross-process safety" gap in `PersistenceManager` (see FileLock.swift).
///
/// These tests use TWO independent `FileLock` instances pointing at the SAME path —
/// each opens its own file descriptor via a fresh `open()` call, so `flock`
/// contention between them is real kernel-level contention (a different "open file
/// description" per POSIX terms), even though both live in this one test process.
/// This is the same contention pattern that would occur between two real OS
/// processes, just without needing to actually spawn a second process in CI.
final class FileLockTests: XCTestCase {

    private func makeTempLockPath() throws -> (dir: URL, lockPath: URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "FileLockTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return (dir, dir.appendingPathComponent("test.lock"))
    }

    func testExclusiveLockBlocksSecondExclusiveLock() throws {
        let (dir, lockPath) = try makeTempLockPath()
        defer { try? FileManager.default.removeItem(at: dir) }

        let lockA = try FileLock(path: lockPath)
        let lockB = try FileLock(path: lockPath)

        let heldFor: TimeInterval = 0.3
        let start = Date()
        let releasedExpectation = expectation(description: "lockA released")

        DispatchQueue.global().async {
            try? lockA.withLock(exclusive: true) {
                Thread.sleep(forTimeInterval: heldFor)
            }
            releasedExpectation.fulfill()
        }

        Thread.sleep(forTimeInterval: 0.05)

        try lockB.withLock(exclusive: true) {
            // Reaching here means lockB blocked until lockA released.
        }

        let elapsed = Date().timeIntervalSince(start)
        XCTAssertGreaterThanOrEqual(
            elapsed, heldFor - 0.05,
            "lockB should have blocked until lockA released the exclusive lock")

        wait(for: [releasedExpectation], timeout: 2.0)
    }

    /// The two tests above cover exclusive-vs-exclusive and shared-vs-shared,
    /// but not the third combination a reader/writer-style lock must also
    /// honor: a shared (`LOCK_SH`) request must still block while an
    /// exclusive (`LOCK_EX`) lock is held elsewhere. Without this, the
    /// contract isn't actually proven — only two of its three cases are.
    func testExclusiveLockBlocksSharedLock() throws {
        let (dir, lockPath) = try makeTempLockPath()
        defer { try? FileManager.default.removeItem(at: dir) }

        let lockA = try FileLock(path: lockPath)
        let lockB = try FileLock(path: lockPath)

        let heldFor: TimeInterval = 0.3
        let start = Date()
        let releasedExpectation = expectation(description: "lockA released")

        DispatchQueue.global().async {
            try? lockA.withLock(exclusive: true) {
                Thread.sleep(forTimeInterval: heldFor)
            }
            releasedExpectation.fulfill()
        }

        Thread.sleep(forTimeInterval: 0.05)

        try lockB.withLock(exclusive: false) {
            // Reaching here means lockB (shared) blocked until lockA (exclusive) released.
        }

        let elapsed = Date().timeIntervalSince(start)
        XCTAssertGreaterThanOrEqual(
            elapsed, heldFor - 0.05,
            "A shared lock request should have blocked until the exclusive lock was released")

        wait(for: [releasedExpectation], timeout: 2.0)
    }

    func testSharedLocksDoNotBlockEachOther() throws {
        let (dir, lockPath) = try makeTempLockPath()
        defer { try? FileManager.default.removeItem(at: dir) }

        let lockA = try FileLock(path: lockPath)
        let lockB = try FileLock(path: lockPath)

        let bothAcquired = expectation(description: "both shared locks held concurrently")
        bothAcquired.expectedFulfillmentCount = 2

        let start = Date()
        for lock in [lockA, lockB] {
            DispatchQueue.global().async {
                try? lock.withLock(exclusive: false) {
                    Thread.sleep(forTimeInterval: 0.2)
                }
                bothAcquired.fulfill()
            }
        }

        wait(for: [bothAcquired], timeout: 1.0)

        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(
            elapsed, 0.35,
            "Two shared (LOCK_SH) locks should run concurrently, not serialize")
    }

    func testWithLockReleasesOnThrow() throws {
        let (dir, lockPath) = try makeTempLockPath()
        defer { try? FileManager.default.removeItem(at: dir) }

        struct DummyError: Error {}
        let lockA = try FileLock(path: lockPath)
        let lockB = try FileLock(path: lockPath)

        XCTAssertThrowsError(
            try lockA.withLock(exclusive: true) {
                throw DummyError()
            })

        let acquired = expectation(description: "lockB acquired after lockA threw")
        DispatchQueue.global().async {
            try? lockB.withLock(exclusive: true) {}
            acquired.fulfill()
        }
        wait(for: [acquired], timeout: 1.0)
    }
}
