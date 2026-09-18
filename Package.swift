// swift-tools-version:5.9
import PackageDescription

// DEPENDENCY POLICY DECISION (recorded per guide §2):
// We are using a pure-Swift, zero-dependency binary heap implementation
// (BinaryHeap.swift, Utilities/).
// Rationale: maximum portability, zero supply-chain risk, no external fetch
// required for CI, and the implementation is straightforward. The alternative
// (apple/swift-collections Heap) is pure Swift but adds a dependency we don't
// need. This decision is final — do not introduce swift-collections later
// without revisiting this comment.
//
// BUILD FLAGS POLICY (fix Y2):
// This package intentionally declares NO `.unsafeFlags(...)`. SwiftPM refuses to
// resolve a package that contains unsafe build flags when it is consumed as a
// versioned dependency ("the target 'VectorDatabase' ... contains unsafe build flags"),
// which would make this library impossible to add via
// `.package(url:from:)`. `-Ounchecked` previously lived here; it is now an
// OPT-IN decision for the consumer instead:
//
//     swift build  -c release -Xswiftc -Ounchecked
//     swift test   -c release -Xswiftc -Ounchecked
//     swift run    -c release -Xswiftc -Ounchecked VectorDatabaseBenchmarks
//
// Everything in the library is written to be correct WITHOUT -Ounchecked; the
// flag only removes bounds/overflow traps for extra speed in the hot float
// loops. Note that `precondition`/`assert` are stripped under -Ounchecked, which
// is why the few checks that must survive a release build (file header size,
// see PersistenceManager) use `fatalError` rather than `precondition`.

let package = Package(
    name: "swift-vector-database",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        .library(name: "VectorDatabase", targets: ["VectorDatabase"])
    ],
    dependencies: [
        // Dependency policy: zero-dependency path chosen (see comment above).
    ],
    targets: [
        .target(
            name: "VectorDatabase",
            dependencies: []
        ),
        .testTarget(
            name: "VectorDatabaseTests",
            dependencies: ["VectorDatabase"]
        ),
        // Standalone benchmark executable — NOT XCTest.
        // Lives in Benchmarks/VectorDatabaseBenchmarks/main.swift.
        .executableTarget(
            name: "VectorDatabaseBenchmarks",
            dependencies: ["VectorDatabase"],
            path: "Benchmarks/VectorDatabaseBenchmarks"
        ),
    ]
)
