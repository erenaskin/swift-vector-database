/// VisitedListTests.swift — Unit tests for the epoch-based VisitedList tracker.
///
/// Tests are grouped into four categories:
///   [VL-1] Basic correctness: insert / contains within a single epoch.
///   [VL-2] Epoch-0 collision safety: fresh list must not report any ID as visited.
///   [VL-3] Epoch reset isolation: IDs from a previous epoch become invisible after nextEpoch().
///   [VL-4] UInt16.max overflow: rollover resets state correctly, no stale false-positives.

import XCTest
@testable import VectorDatabase

final class VisitedListTests: XCTestCase {

    // MARK: - [VL-1] Basic Correctness

    func testInsertAndContainsWithinOneEpoch() {
        var vl = VisitedList(capacity: 100)
        vl.nextEpoch()

        XCTAssertFalse(vl.contains(0))
        XCTAssertFalse(vl.contains(42))
        XCTAssertFalse(vl.contains(99))

        vl.insert(42)
        XCTAssertTrue(vl.contains(42))
        XCTAssertFalse(vl.contains(0))
        XCTAssertFalse(vl.contains(99))

        vl.insert(0)
        vl.insert(99)
        XCTAssertTrue(vl.contains(0))
        XCTAssertTrue(vl.contains(99))
    }

    func testWordBoundary_bit63and64and65() {
        // Force-test IDs at exact UInt word boundaries: 63, 64, 65
        var vl = VisitedList(capacity: 128)
        vl.nextEpoch()

        for id: Int32 in [63, 64, 65] {
            XCTAssertFalse(vl.contains(id), "ID \(id) should not be visited before insert")
            vl.insert(id)
            XCTAssertTrue(vl.contains(id), "ID \(id) should be visited after insert")
        }
    }

    func testOutOfBoundsIDDoesNotCrash() {
        // IDs beyond capacity must be silently ignored — no crash, no false positive.
        var vl = VisitedList(capacity: 10)
        vl.nextEpoch()

        vl.insert(999)                       // OOB insert: no-op
        XCTAssertFalse(vl.contains(999))     // OOB contains: always false
        XCTAssertFalse(vl.contains(-1))      // Negative ID: always false
    }

    // MARK: - [VL-2] Epoch-0 Collision Safety

    func testFreshListAfterFirstNextEpochHasNoVisitedIDs() {
        // CRITICAL: The epochs array is zero-initialised. currentEpoch starts at 1.
        // After the first nextEpoch() call (which brings epoch to 2), no ID should
        // be visible as visited. This guards against the "epoch=0 collision" bug where
        // zero-initialised slots would falsely match a zero currentEpoch.
        var vl = VisitedList(capacity: 500)
        vl.nextEpoch()  // epoch becomes 2 (was 1 at init, incremented once)

        for id in stride(from: Int32(0), to: 500, by: 7) {
            XCTAssertFalse(vl.contains(id), "ID \(id) should not be pre-visited")
        }
    }

    func testImmediatelyAfterInitNoIDIsVisited() {
        // Before any nextEpoch() call (epoch = 1, array = all 0s), nothing is visited.
        let vl = VisitedList(capacity: 200)
        for id in Int32(0)..<200 {
            XCTAssertFalse(vl.contains(id))
        }
    }

    // MARK: - [VL-3] Epoch Reset Isolation

    func testNextEpochClearsPreviouslyVisitedIDs() {
        var vl = VisitedList(capacity: 50)
        vl.nextEpoch()

        vl.insert(10)
        vl.insert(20)
        vl.insert(30)
        XCTAssertTrue(vl.contains(10))

        // Advance to next epoch — previous inserts must be invisible.
        vl.nextEpoch()
        XCTAssertFalse(vl.contains(10), "ID 10 should not be visible in next epoch")
        XCTAssertFalse(vl.contains(20))
        XCTAssertFalse(vl.contains(30))
    }

    func testMultipleEpochsDoNotInterfere() {
        var vl = VisitedList(capacity: 20)

        for _ in 0..<10 {
            vl.nextEpoch()
            vl.insert(5)
            vl.insert(15)
            XCTAssertTrue(vl.contains(5))
            XCTAssertTrue(vl.contains(15))
            XCTAssertFalse(vl.contains(0))
        }
    }

    // MARK: - [VL-4] UInt16.max Overflow Safety

    func testOverflowRolloverResetsStateCorrectly() {
        // Manually drive currentEpoch to UInt16.max and verify that:
        //   a) After rollover, no previously-inserted ID is falsely "visited".
        //   b) New inserts in the post-rollover epoch work correctly.
        var vl = VisitedList(capacity: 100)

        // Insert ID 42 in an early epoch.
        vl.nextEpoch()   // epoch = 2
        vl.insert(42)
        XCTAssertTrue(vl.contains(42))

        // Advance through 65,533 more epochs to hit UInt16.max territory.
        // We use a loop; nextEpoch() is O(1) until overflow, then O(capacity) once.
        // capacity=100 means overflow reset is fast.
        for _ in 0..<(Int(UInt16.max) - 2) {
            vl.nextEpoch()
        }
        // We are now at UInt16.max. One more call triggers the overflow reset.
        vl.nextEpoch()   // triggers full array zeroing, epoch = 1

        // ID 42 was inserted in epoch 2. After rollover the array is zeroed,
        // so epochs[42] == 0 != 1 (currentEpoch). Must NOT be visible.
        XCTAssertFalse(vl.contains(42), "ID 42 must not be visible after overflow rollover")

        // Fresh inserts in the post-rollover epoch must work normally.
        vl.insert(7)
        XCTAssertTrue(vl.contains(7))
        XCTAssertFalse(vl.contains(42))
    }

    func testCapacityZeroDoesNotCrash() {
        var vl = VisitedList(capacity: 0)
        vl.nextEpoch()
        vl.insert(0)
        XCTAssertFalse(vl.contains(0))
    }
}
