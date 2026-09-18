import XCTest

@testable import VectorDatabase

/// HNSWParametersValidationTests.swift
///
/// WHY THIS FILE EXISTS:
/// The coverage report for `Public/HNSWParameters.swift` showed only 66.67%
/// region / 64.29% line coverage despite `validate()` having 100% FUNCTION
/// coverage. That combination means the function was always CALLED (every
/// `VectorDatabase.init` calls it) but every one of its `guard ... else { throw }`
/// branches was dead as far as the test suite could tell — including the
/// `seed != 0` check, since nothing in the suite ever constructed an invalid
/// `HNSWParameters` value. A regression in any of these four checks (a typo'd
/// operator, an accidentally-deleted guard) would have shipped silently.
///
/// These tests exercise every branch of `validate()` directly, plus the
/// end-to-end path through `VectorDatabase.init` for the one check (`seed`) whose
/// failure mode used to be a process-killing `precondition` trap deep inside
/// `SeedableRNG` rather than a catchable error — see the FIX comment on
/// `HNSWParameters.validate()`.
final class HNSWParametersValidationTests: XCTestCase {

    // MARK: - Valid configurations

    func testDefaultParametersValidate() {
        XCTAssertNoThrow(try HNSWParameters.default.validate())
    }

    func testCustomValidParametersValidate() {
        let params = HNSWParameters(M: 32, efConstruction: 300, efSearch: 200, seed: 0xC0FFEE)
        XCTAssertNoThrow(try params.validate())
    }

    // MARK: - M

    func testMOfOneThrows() {
        let params = HNSWParameters(M: 1, efConstruction: 100, efSearch: 50, seed: 1)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    func testMOfZeroThrows() {
        let params = HNSWParameters(M: 0, efConstruction: 100, efSearch: 50, seed: 1)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    func testNegativeMThrows() {
        let params = HNSWParameters(M: -4, efConstruction: 100, efSearch: 50, seed: 1)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    // MARK: - efConstruction

    func testZeroEfConstructionThrows() {
        let params = HNSWParameters(M: 16, efConstruction: 0, efSearch: 50, seed: 1)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    func testNegativeEfConstructionThrows() {
        let params = HNSWParameters(M: 16, efConstruction: -10, efSearch: 50, seed: 1)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    // MARK: - efSearch

    func testZeroEfSearchThrows() {
        let params = HNSWParameters(M: 16, efConstruction: 100, efSearch: 0, seed: 1)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    func testNegativeEfSearchThrows() {
        let params = HNSWParameters(M: 16, efConstruction: 100, efSearch: -1, seed: 1)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    // MARK: - seed

    func testZeroSeedThrowsFromValidate() {
        let params = HNSWParameters(M: 16, efConstruction: 100, efSearch: 50, seed: 0)
        XCTAssertThrowsError(try params.validate()) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }

    /// End-to-end regression test for the bug this file was added to guard
    /// against: `VectorDatabase.init` must reject `seed: 0` with a normal, catchable
    /// `VectorDatabaseError.invalidParameters` — NOT crash the process later inside
    /// `SeedableRNG.init`'s `precondition`. Before the `seed != 0` guard was
    /// added to `validate()`, this line would have thrown nothing here and
    /// instead trapped much later, e.g. on whichever `insert()` call happened
    /// to cross `hnswThreshold` and trigger the FlatIndex → HNSWIndex
    /// migration (or immediately, when loading a persisted HNSW snapshot).
    func testVectorDatabaseInitRejectsZeroSeedInsteadOfCrashingLater() throws {
        let badParams = HNSWParameters(seed: 0)
        XCTAssertThrowsError(
            try VectorDatabase(dimension: 4, metric: .cosine, parameters: badParams)
        ) { error in
            guard case VectorDatabaseError.invalidParameters = error else {
                return XCTFail("Expected .invalidParameters, got \(error)")
            }
        }
    }
}
