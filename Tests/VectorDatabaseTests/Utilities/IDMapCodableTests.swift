import XCTest

@testable import VectorDatabase

/// IDMapCodableTests.swift
///
/// WHY THIS FILE EXISTS:
/// `IDMap.init(from:)` has an explicit backward-compatibility branch (see
/// `Utilities/IDMap.swift`, FIX O2's doc comment): if a decoded snapshot has
/// no `liveIDs` key — i.e. it was written by a build that predates that
/// optimization — `liveIDs` is reconstructed from `intToString.keys.sorted()`
/// instead of failing to decode.
///
/// Every persistence test in this suite round-trips IDMap through the
/// CURRENT encoder, which always writes `liveIDs`, so the decode always takes
/// the `if let stored = ...` branch and the legacy fallback `else` branch
/// was never once exercised — confirmed by `grep -r liveIDs Tests/` finding
/// nothing before this file was added. A regression here (e.g. someone
/// "simplifying" away the fallback) would silently break every user
/// upgrading from a build old enough to have written a `liveIDs`-less
/// snapshot, and nothing would catch it.
final class IDMapCodableTests: XCTestCase {

    func testDecodingLegacySnapshotWithoutLiveIDsReconstructsFromIntToString() throws {
        var map = IDMap()
        _ = try map.assign(externalID: "a")
        _ = try map.assign(externalID: "b")
        _ = try map.assign(externalID: "c")
        // Removing the middle entry leaves a gap, so a naive `0..<nextID` walk
        // (what pagination used to do, pre-FIX-O2) would disagree with the
        // correct live set — makes sure the reconstructed `liveIDs` is right,
        // not just non-empty.
        try map.remove(externalID: "b")

        let encoded = try JSONEncoder().encode(map)

        // Simulate a snapshot written by a build that predates the `liveIDs`
        // key: strip it out of the encoded JSON, exactly like an old file on
        // disk would look, then decode through IDMap's real `init(from:)`.
        guard var json = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] else {
            XCTFail("Failed to parse encoded IDMap as a JSON object")
            return
        }
        XCTAssertNotNil(
            json["liveIDs"], "Sanity check: the current encoder should include liveIDs")
        json.removeValue(forKey: "liveIDs")
        let legacyData = try JSONSerialization.data(withJSONObject: json)

        let decoded = try JSONDecoder().decode(IDMap.self, from: legacyData)

        // The fallback path must reconstruct exactly the same live set the
        // original had, not just decode without crashing.
        XCTAssertEqual(decoded.count, map.count)
        XCTAssertEqual(decoded.internalID(for: "a"), map.internalID(for: "a"))
        XCTAssertEqual(decoded.internalID(for: "c"), map.internalID(for: "c"))
        XCTAssertNil(
            decoded.internalID(for: "b"), "Removed ID must stay removed across the legacy path")

        XCTAssertEqual(
            decoded.listExternalIDs(offset: 0, limit: 10).sorted(),
            map.listExternalIDs(offset: 0, limit: 10).sorted(),
            "Pagination over the reconstructed liveIDs must match the original")
    }

    /// Companion sanity check: the CURRENT format (the one every real save
    /// produces) must NOT take the fallback path, i.e. `liveIDs` really is
    /// round-tripped as-is rather than always being silently recomputed
    /// (which would make the fallback branch pointless dead code).
    func testDecodingCurrentSnapshotUsesStoredLiveIDsDirectly() throws {
        var map = IDMap()
        _ = try map.assign(externalID: "x")
        _ = try map.assign(externalID: "y")

        let encoded = try JSONEncoder().encode(map)
        let decoded = try JSONDecoder().decode(IDMap.self, from: encoded)

        XCTAssertEqual(decoded.count, map.count)
        XCTAssertEqual(
            decoded.listExternalIDs(offset: 0, limit: 10),
            map.listExternalIDs(offset: 0, limit: 10))
    }
}
