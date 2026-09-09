// swift-tools-version:5.9
import PackageDescription

// DEPENDENCY POLICY DECISION (recorded per guide §2):
// We are using a pure-Swift, zero-dependency binary heap implementation
// (BinaryHeap.swift, ~80 lines, to be written in Phase 4 in Utilities/).
// Rationale: maximum portability, zero supply-chain risk, no external fetch
// required for CI, and the implementation is straightforward. The alternative
// (apple/swift-collections Heap) is pure Swift but adds a dependency we don't
// need. This decision is final — do not introduce swift-collections later
// without revisiting this comment.
//
// NOTE: -Ounchecked is gated to release config ONLY and must NOT be considered
// active or relied upon until Phase 1's correctness tests are green. While
// debugging, you want crashes, not silent corruption.

let package = Package(
    name: "SwiftVectorDB",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(name: "VectorDB", targets: ["VectorDB"]),
    ],
    dependencies: [
        // Dependency policy: zero-dependency path chosen (see comment above).
        // swift-collections is kept here as a reference but commented out.
        // If the policy ever changes, uncomment the line below AND update the
        // comment at the top of this file before Phase 4 begins.
        // .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "VectorDB",
            dependencies: [
                // "Collections", .product(name: "Collections", package: "swift-collections")
                // Uncomment above only if dependency policy changes to swift-collections.
            ],
            swiftSettings: [
                // -Ounchecked: disables array bounds/overflow checks in release builds.
                // This gives a meaningful speedup once float operations are in the millions.
                // DO NOT enable until Phase 1 correctness tests are green — you want
                // trapping crashes during development, not silent memory corruption.
                .unsafeFlags(["-Ounchecked"], .when(configuration: .release))
            ]
        ),
        .testTarget(
            name: "VectorDBTests",
            dependencies: ["VectorDB"]
        ),
        // Standalone benchmark executable — NOT XCTest.
        // Lives in Benchmarks/VectorDBBenchmarks/main.swift.
        // Implementation: Phase 10 (benchmarking & tuning).
        .executableTarget(
            name: "VectorDBBenchmarks",
            dependencies: ["VectorDB"],
            path: "Benchmarks/VectorDBBenchmarks"
        ),
    ]
)
