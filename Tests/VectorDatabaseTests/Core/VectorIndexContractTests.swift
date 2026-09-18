/// VectorIndexContractTests.swift — Behavioral contract tests for the shared
/// `VectorIndex` protocol (Core/IndexProtocol.swift).
///
/// FlatIndex and HNSWIndex both conform to `VectorIndex`, but nothing in the
/// codebase actually exercised them polymorphically through it: `IndexRouter`
/// dispatches via a closed `enum` (for performance, avoiding existential
/// overhead) rather than `any VectorIndex`, and every other test in this
/// suite constructs and calls the concrete types directly. That left
/// `VectorIndex` as pure documentation, with nothing verifying that both
/// conforming types actually honor the same observable contract.
///
/// These tests run an identical, small script of protocol calls against both
/// concrete types through a generic `some VectorIndex` parameter, and assert
/// they agree on every operation the protocol promises is shared behavior.
/// (HNSW's approximate search is not required to agree with FlatIndex's exact
/// brute force in general — but for a handful of well-separated, orthogonal
/// points with a generous `ef`, it must, and the exact-match behaviors like
/// `getVector`/`remove`/`count` must agree exactly in all cases.)
import XCTest
@testable import VectorDatabase

final class VectorIndexContractTests: XCTestCase {

    private struct ContractResult {
        let countAfterInsert: Int
        let selfSearchIDs: [Int32]
        let vectorBeforeRemove: [Float]?
        let countAfterRemove: Int
        let vectorAfterRemove: [Float]?
        let searchAfterRemoveIDs: [Int32]
    }

    /// Runs a small, well-separated dataset through any `VectorIndex`
    /// conformer and returns the results of a shared script of operations, so
    /// two concrete types can be compared against identical inputs.
    private func runContractScript<Index: VectorIndex>(_ makeIndex: () -> Index) throws -> ContractResult {
        var index = makeIndex()

        // Four axis-aligned unit vectors: each is trivially its own exact
        // nearest neighbor under the dotProduct metric (dot with itself = 1,
        // dot with any other = 0), so both an exact and an approximate index
        // must agree on the top-1 result.
        let vectors: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        for (i, v) in vectors.enumerated() {
            try v.withUnsafeBufferPointer { buf in
                try index.insert(internalID: Int32(i), vector: buf.baseAddress!)
            }
        }

        let countAfterInsert = index.count

        let selfSearchIDs: [Int32] = vectors[0].withUnsafeBufferPointer { buf in
            index.search(query: buf.baseAddress!, k: 1, ef: 64).map { $0.id }
        }

        let vectorBeforeRemove = index.getVector(internalID: 1)

        try index.remove(internalID: 1)
        let countAfterRemove = index.count
        let vectorAfterRemove = index.getVector(internalID: 1)

        // After removing id 1, querying with its old vector must never
        // return id 1 again, no matter how large k is.
        let searchAfterRemoveIDs: [Int32] = vectors[1].withUnsafeBufferPointer { buf in
            index.search(query: buf.baseAddress!, k: vectors.count, ef: 64).map { $0.id }
        }

        return ContractResult(
            countAfterInsert: countAfterInsert,
            selfSearchIDs: selfSearchIDs,
            vectorBeforeRemove: vectorBeforeRemove,
            countAfterRemove: countAfterRemove,
            vectorAfterRemove: vectorAfterRemove,
            searchAfterRemoveIDs: searchAfterRemoveIDs)
    }

    func testFlatIndexAndHNSWIndexHonorTheSameVectorIndexContract() throws {
        let dim = 4

        let flatResult = try runContractScript {
            FlatIndex(dimension: dim, metric: .dotProduct)
        }

        let hnswResult = try runContractScript {
            HNSWIndex(
                dimension: dim, metric: .dotProduct,
                params: HNSWParameters(M: 4, efConstruction: 32, efSearch: 32, seed: 0x1234))
        }

        XCTAssertEqual(flatResult.countAfterInsert, 4)
        XCTAssertEqual(hnswResult.countAfterInsert, 4)

        XCTAssertEqual(flatResult.selfSearchIDs.first, 0,
            "FlatIndex: querying with vector 0 exactly must return id 0 as the top hit")
        XCTAssertEqual(hnswResult.selfSearchIDs.first, 0,
            "HNSWIndex: querying with vector 0 exactly must return id 0 as the top hit")

        XCTAssertEqual(flatResult.vectorBeforeRemove, [0, 1, 0, 0])
        XCTAssertEqual(hnswResult.vectorBeforeRemove, [0, 1, 0, 0])

        XCTAssertEqual(flatResult.countAfterRemove, 3)
        XCTAssertEqual(hnswResult.countAfterRemove, 3)

        XCTAssertNil(flatResult.vectorAfterRemove,
            "FlatIndex: getVector for a removed id must return nil")
        XCTAssertNil(hnswResult.vectorAfterRemove,
            "HNSWIndex: getVector for a removed id must return nil")

        XCTAssertFalse(flatResult.searchAfterRemoveIDs.contains(1),
            "FlatIndex: a removed id must never reappear in search results")
        XCTAssertFalse(hnswResult.searchAfterRemoveIDs.contains(1),
            "HNSWIndex: a removed id must never reappear in search results")
    }

    /// Compile-time-adjacent check that both concrete types are actually
    /// usable as `any VectorIndex` existentials, not just structurally
    /// conforming. If either type stopped conforming to `VectorIndex`, this
    /// file would fail to compile — which is the point: `VectorIndex` is
    /// meant to be a real, usable interface, not documentation that happens
    /// to type-check.
    func testBothIndexTypesAreUsableAsVectorIndexExistential() throws {
        let dim = 2
        var indexes: [any VectorIndex] = [
            FlatIndex(dimension: dim, metric: .dotProduct),
            HNSWIndex(dimension: dim, metric: .dotProduct),
        ]

        for i in indexes.indices {
            let vector: [Float] = [1, 0]
            try vector.withUnsafeBufferPointer { buf in
                try indexes[i].insert(internalID: 0, vector: buf.baseAddress!)
            }
            XCTAssertEqual(indexes[i].count, 1)
        }
    }
}
