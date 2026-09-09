# Vector Database with Swift

A zero-dependency, pure Swift, high-performance Vector Database tailored for on-device machine learning (CoreML) workloads on iOS and macOS.

Built specifically to bring **blazingly fast similarity search** to Apple devices without the overhead of C++ interop or heavy third-party SDKs. It uses a **Hierarchical Navigable Small World (HNSW)** graph index combined with Memory-Mapped (mmap) storage to handle massive vector embeddings using minimal RAM.

---

## Key Features

- **HNSW Indexing**: Delivers sub-millisecond approximate nearest neighbor searches with near-perfect recall (99.9% at 50k vectors).
- **Hardware Accelerated**: Leverages Apple's `Accelerate` (vDSP) framework for blazing-fast SIMD vector math optimized exclusively for Apple Silicon.
- **Local & Privacy First**: 100% on-device. No network calls, no cloud dependencies. Your users' data never leaves their device.
- **Zero Dependencies**: Pure Swift codebase. It compiles incredibly fast and adds minimal binary size overhead to your iOS apps.
- **Thread-safe by Design**: Safe concurrent access powered by Swift `actor` semantics and internal low-level Reader-Writer locks.
- **Memory Efficient (mmap)**: Backed by Memory-Mapped Files, allowing the OS to seamlessly page large vector datasets in and out of RAM under memory pressure.

---

## Quick Start

Initialize the database, insert your embeddings, and search seamlessly:

```swift
import VectorDB

// 1. Initialize the database
// Set path to `nil` for purely in-memory execution!
let storeURL = URL(fileURLWithPath: "/path/to/storage")
let db = try VectorDB(dimension: 384, metric: .cosine, path: storeURL)

// 2. Insert vectors (with optional metadata)
let embedding: [Float] = [0.12, 0.45, -0.89, /* ... */]
try await db.insert(id: "doc_1", vector: embedding, metadata: ["title": "Swift Guide"])

// 3. Search for nearest neighbors
let query: [Float] = [0.10, 0.40, -0.90, /* ... */]
let results = try await db.search(query: query, k: 5)

for result in results {
    print("Found \(result.id) with score \(result.score)")
}
```

---

## Energy Profiling (Instruments)

Because VectorDB is built for iOS devices, **battery consumption** is just as critical as raw CPU speed. 

We provide a built-in SwiftUI `EnergyProfilerView` under the `Benchmarks/` directory so you can run Apple's **Instruments** and measure the exact energy impact on a real device.

### How to Profile Energy:
1. Open the project in Xcode.
2. Select the **VectorDBBenchmarks** target (or integrate `EnergyProfilerView` into an iOS app).
3. Connect a physical iPhone/iPad (Energy profiling does not work on the Simulator).
4. Go to **Product > Profile** (⌘I) in Xcode.
5. In Instruments, choose the **Energy Log** template.
6. When the app launches, tap **"Start 15s Workload"**.
7. Instruments will track the exact power overhead (CPU, overhead, and thermal state) of extreme continuous inserts and searches. 

*See the [Benchmarks Report](BENCHMARKS.md) for our measured baseline numbers!*

---

## Documentation & Internals

For a deep dive into the architecture, memory mapping, graph topology, and internal design decisions, please read our comprehensive [Vector Database Documentation](vector-database-documentation.md).

For detailed performance characteristics, latency measurements, and exact benchmarks against Flat Indexing, see the [Benchmarks Report](BENCHMARKS.md).

---

## Semantic Versioning Policy

`VectorDB` adheres strictly to Semantic Versioning (SemVer), particularly regarding data persistence to guarantee your app updates never break user data.

- **Major versions (`x.0.0`)**: Indicate a breaking change to the **on-disk binary format**. 
  - The on-disk header contains a `formatVersion`. When the SDK updates to a new major version, it may not be able to silently read older files without a manual migration step. This guarantees that SDK updates will **never silently corrupt** your users' local disk stores. 
- **Minor/Patch versions (`1.x.x`)**: Are guaranteed to be fully backwards-compatible at the storage layer.

---