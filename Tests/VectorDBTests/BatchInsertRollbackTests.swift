import XCTest
@testable import VectorDB

final class BatchInsertRollbackTests: XCTestCase {
    
    func testRollbackOnPreExistingDuplicateID() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct)
        
        // Pre-insert one record
        try await db.insert(id: "existing_id", vector: [1.0, 1.0])
        let preBatchCount = await db.stats().liveCount
        XCTAssertEqual(preBatchCount, 1)
        
        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            ("new_1", [1.0, 1.0], nil),
            ("new_2", [1.0, 1.0], nil),
            ("existing_id", [1.0, 1.0], nil), // Duplicate! Will throw
            ("new_4", [1.0, 1.0], nil),
            ("new_5", [1.0, 1.0], nil)
        ]
        
        do {
            try await db.batchInsert(batch)
            XCTFail("Should have thrown duplicateID")
        } catch VectorDBError.duplicateID(let id) {
            XCTAssertEqual(id, "existing_id")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
        
        let postBatchCount = await db.stats().liveCount
        XCTAssertEqual(postBatchCount, preBatchCount, "Items inserted before the failure must have been rolled back")
    }
    
    func testRollbackOnSameBatchDuplicateID() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct)
        let preBatchCount = await db.stats().liveCount
        XCTAssertEqual(preBatchCount, 0)
        
        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            ("new_1", [1.0, 1.0], nil),
            ("new_2", [1.0, 1.0], nil),
            ("new_1", [1.0, 1.0], nil), // Duplicate of item 0! Will throw during the insert loop
            ("new_4", [1.0, 1.0], nil)
        ]
        
        do {
            try await db.batchInsert(batch)
            XCTFail("Should have thrown duplicateID")
        } catch VectorDBError.duplicateID(let id) {
            XCTAssertEqual(id, "new_1")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
        
        let postBatchCount = await db.stats().liveCount
        XCTAssertEqual(postBatchCount, preBatchCount, "All items in the failed batch must have been rolled back")
    }
    
    func testFullyValidBatchSuccess() async throws {
        let db = try VectorDB(dimension: 2, metric: .dotProduct)
        let preBatchCount = await db.stats().liveCount
        XCTAssertEqual(preBatchCount, 0)
        
        let batch: [(id: String, vector: [Float], metadata: [String: String]?)] = [
            ("new_1", [1.0, 1.0], nil),
            ("new_2", [1.0, 1.0], nil),
            ("new_3", [1.0, 1.0], nil)
        ]
        
        try await db.batchInsert(batch)
        
        let postBatchCount = await db.stats().liveCount
        XCTAssertEqual(postBatchCount, 3, "Valid batch should insert all items successfully")
    }
}
