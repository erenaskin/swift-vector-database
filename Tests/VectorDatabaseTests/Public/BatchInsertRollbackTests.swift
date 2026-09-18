import XCTest

@testable import VectorDatabase

final class BatchInsertRollbackTests: XCTestCase {

    func testRollbackOnPreExistingDuplicateID() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)

        // Pre-insert one record
        try await db.insert(id: "existing_id", vector: [1.0, 1.0])
        let preBatchCount = await db.stats().liveCount
        XCTAssertEqual(preBatchCount, 1)

        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            ("new_1", [1.0, 1.0], nil),
            ("new_2", [1.0, 1.0], nil),
            ("existing_id", [1.0, 1.0], nil),  // Duplicate! Will throw
            ("new_4", [1.0, 1.0], nil),
            ("new_5", [1.0, 1.0], nil),
        ]

        do {
            try await db.batchInsert(batch)
            XCTFail("Should have thrown duplicateID")
        } catch VectorDatabaseError.duplicateID(let id) {
            XCTAssertEqual(id, "existing_id")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let postBatchCount = await db.stats().liveCount
        XCTAssertEqual(
            postBatchCount, preBatchCount,
            "Items inserted before the failure must have been rolled back")
    }

    func testRollbackOnSameBatchDuplicateID() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        let preBatchCount = await db.stats().liveCount
        XCTAssertEqual(preBatchCount, 0)

        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            ("new_1", [1.0, 1.0], nil),
            ("new_2", [1.0, 1.0], nil),
            ("new_1", [1.0, 1.0], nil),  // Duplicate of item 0! Will throw during the insert loop
            ("new_4", [1.0, 1.0], nil),
        ]

        do {
            try await db.batchInsert(batch)
            XCTFail("Should have thrown duplicateID")
        } catch VectorDatabaseError.duplicateID(let id) {
            XCTAssertEqual(id, "new_1")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let postBatchCount = await db.stats().liveCount
        XCTAssertEqual(
            postBatchCount, preBatchCount,
            "All items in the failed batch must have been rolled back")
    }

    /// The two tests above both fail partway *through* the batch (after at
    /// least one item has already been inserted), so `insertedIDs` is
    /// non-empty when the `catch` block's reverse-order rollback loop runs.
    /// This test covers the boundary the other two don't: the very FIRST
    /// item is the duplicate, so `insertedIDs` is still empty at rollback
    /// time and the rollback loop's body never executes. This is worth
    /// pinning down on its own — an off-by-one in the rollback loop (e.g.
    /// assuming at least one prior insert) would only show up here.
    func testRollbackWhenFirstItemIsDuplicate() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)

        try await db.insert(id: "existing_id", vector: [1.0, 1.0])
        let preBatchCount = await db.stats().liveCount
        XCTAssertEqual(preBatchCount, 1)

        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            ("existing_id", [1.0, 1.0], nil),  // Duplicate on the very first item.
            ("new_1", [1.0, 1.0], nil),
            ("new_2", [1.0, 1.0], nil),
        ]

        do {
            try await db.batchInsert(batch)
            XCTFail("Should have thrown duplicateID")
        } catch VectorDatabaseError.duplicateID(let id) {
            XCTAssertEqual(id, "existing_id")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let postBatchCount = await db.stats().liveCount
        XCTAssertEqual(
            postBatchCount, preBatchCount,
            "Failing on the first item must leave the pre-existing record untouched, "
                + "and must not attempt to insert or roll back anything else")

        // The items after the duplicate must never have been attempted at all.
        let new1 = await db.get(id: "new_1")
        let new2 = await db.get(id: "new_2")
        XCTAssertNil(new1)
        XCTAssertNil(new2)
    }

    func testFullyValidBatchSuccess() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .dotProduct)
        let preBatchCount = await db.stats().liveCount
        XCTAssertEqual(preBatchCount, 0)

        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            ("new_1", [1.0, 1.0], nil),
            ("new_2", [1.0, 1.0], nil),
            ("new_3", [1.0, 1.0], nil),
        ]

        try await db.batchInsert(batch)

        let postBatchCount = await db.stats().liveCount
        XCTAssertEqual(postBatchCount, 3, "Valid batch should insert all items successfully")
    }
}
