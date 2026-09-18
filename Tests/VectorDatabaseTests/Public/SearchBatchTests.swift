//
//  SearchBatchTests.swift
//  SwiftVectorDatabase
//
//  Created by Eren AŞKIN on 14.09.2026.
//

import XCTest

@testable import VectorDatabase

/// SearchBatchTests.swift — Verifies `VectorDatabase.searchBatch(queries:)` (added to
/// actually exercise `Engine`'s multi-reader `ReadWriteLock` through the public API;
/// see the doc comment on `searchBatch` in `Public/VectorDatabase.swift`).
///
/// The core correctness property under test: running N queries through
/// `searchBatch` must produce EXACTLY the same results, in the same order, as
/// running each query individually through `search(query:)` — the parallel fan-out
/// must be purely a performance change, never an observable behavior change.
final class SearchBatchTests: XCTestCase {

    func testSearchBatchMatchesSequentialSearchDotProduct() async throws {
        let db = try VectorDatabase(dimension: 8, metric: .dotProduct)
        var rng = SeedableRNG(seed: 0x1111_2222_3333_4444)

        for i in 0..<300 {
            let vector = (0..<8).map { _ in rng.nextFloat() }
            try await db.insert(id: "id_\(i)", vector: vector)
        }

        let queries: [[Float]] = (0..<25).map { _ in (0..<8).map { _ in rng.nextFloat() } }

        let batchResults = try await db.searchBatch(queries: queries, k: 5)
        XCTAssertEqual(batchResults.count, queries.count)

        for (i, query) in queries.enumerated() {
            let sequential = try await db.search(query: query, k: 5)
            XCTAssertEqual(
                batchResults[i].map(\.id), sequential.map(\.id),
                "searchBatch result order/content must match sequential search() at index \(i)"
            )
            for (b, s) in zip(batchResults[i], sequential) {
                XCTAssertEqual(b.score, s.score, accuracy: 1e-6)
            }
        }
    }

    func testSearchBatchMatchesSequentialSearchCosine() async throws {
        let db = try VectorDatabase(dimension: 16, metric: .cosine)
        var rng = SeedableRNG(seed: 0x5555_6666_7777_8888)

        for i in 0..<500 {
            let vector = (0..<16).map { _ in rng.nextFloat() }
            try await db.insert(id: "cos_\(i)", vector: vector)
        }

        let queries: [[Float]] = (0..<10).map { _ in (0..<16).map { _ in rng.nextFloat() } }

        let batchResults = try await db.searchBatch(queries: queries, k: 3)
        for (i, query) in queries.enumerated() {
            let sequential = try await db.search(query: query, k: 3)
            XCTAssertEqual(batchResults[i].map(\.id), sequential.map(\.id))
        }
    }

    func testSearchBatchEmptyQueriesReturnsEmptyArray() async throws {
        let db = try VectorDatabase(dimension: 4, metric: .euclidean)
        let results = try await db.searchBatch(queries: [], k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchBatchKZeroReturnsEmptyResultsPerQuery() async throws {
        let db = try VectorDatabase(dimension: 4, metric: .euclidean)
        try await db.insert(id: "a", vector: [1, 2, 3, 4])
        let results = try await db.searchBatch(queries: [[1, 2, 3, 4], [0, 0, 0, 1]], k: 0)
        XCTAssertEqual(results.count, 2)
        XCTAssertTrue(results.allSatisfy { $0.isEmpty })
    }

    func testSearchBatchRejectsDimensionMismatch() async throws {
        let db = try VectorDatabase(dimension: 4, metric: .euclidean)
        try await db.insert(id: "a", vector: [1, 2, 3, 4])
        do {
            _ = try await db.searchBatch(queries: [[1, 2, 3, 4], [1, 2, 3]], k: 5)
            XCTFail("Expected dimensionMismatch to be thrown")
        } catch VectorDatabaseError.dimensionMismatch {
            // expected
        }
    }
}
