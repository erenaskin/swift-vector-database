# Pure Swift Vector Database

![Swift 5.9+](https://img.shields.io/badge/Swift-5.9%2B-orange)
![Platforms](https://img.shields.io/badge/Platforms-iOS%2015%2B%20%7C%20macOS%2012%2B-blue)
![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen)

A zero-dependency, pure Swift, high-performance Vector Database tailored for on-device machine learning (CoreML) workloads on iOS and macOS.

**What is a Vector Database?**
In modern AI (like ChatGPT or semantic search), data like text, images, or audio is converted into long arrays of numbers called "vectors" (or embeddings). A vector database stores these numbers and quickly finds the "most similar" ones. This is how AI remembers context, finds similar documents, or recommends products.

This project is built specifically to bring **blazingly fast similarity search** to Apple devices natively. By avoiding C++ interop or heavy third-party SDKs, it keeps your app size small and compiles instantly. It uses a **Hierarchical Navigable Small World (HNSW)** graph index combined with Memory-Mapped (mmap) storage to handle massive datasets using minimal RAM.

---

## Key Features

- **Blazing Fast (HNSW Indexing)**: Achieves sub-millisecond approximate nearest neighbor (ANN) searches with near-perfect recall (1.0 at 10k vectors), delivering **0.032ms p50 latency** (`efSearch=10`) on NLP embeddings.
- **Hardware Accelerated**: Leverages Apple's `Accelerate` framework (`vDSP_dotpr` and `cblas_sgemv`) for highly optimized SIMD vector math, yielding a **~7.6x throughput boost** and **3.56x batch search speedup** over scalar loops.
- **High-Throughput Graph Construction**: Epoch-based VisitedList optimizations enable **295.3 inserts/sec** at a 500k vector scale (a **3.9x speedup** over the FlatIndex baseline).
- **Local & Privacy First**: 100% on-device. No network calls, no cloud dependencies. Your users' sensitive AI data never leaves their device.
- **Zero Dependencies**: A pure Swift codebase. It compiles incredibly fast and adds almost zero binary size overhead to your iOS apps.
- **Crash-Safe Durability (WAL)**: Features a robust Write-Ahead Log (WAL) with per-record checksums. If your app crashes or the battery dies during an insert, the database automatically recovers all vectors and metadata seamlessly on next launch.
- **Memory Efficient (mmap)**: Backed by Memory-Mapped Files, allowing the OS to intelligently page large vector datasets in and out of RAM under memory pressure.
- **Rich Metadata Support**: Store custom `[String: String]` metadata alongside your vectors and retrieve them instantly.
- **Thread-safe by Design**: Public API access via the `VectorDatabase` actor guarantees thread safety and prevents data races.
- **Cross-Process Safety**: Advisory file locking (`flock`) prevents corruption when multiple processes (e.g. app + Share Extension) access the same database.

---

## Installation

### Swift Package Manager

Add the dependency to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/erenaskin/swift-vector-database.git", from: "2.0.0")
]
```

Then add `VectorDatabase` to your target's dependencies:

```swift
.target(
    name: "YourApp",
    dependencies: ["VectorDatabase"]
)
```

Or in Xcode: **File → Add Package Dependencies…** and paste the repository URL.

**Requirements:** iOS 15+ / macOS 12+ · Swift 5.9+ · No external dependencies.

---

## Quick Start

Initialize the database, insert your embeddings (with optional metadata), and search seamlessly.

### 1. Initialize the Database
You can create a purely in-memory database by passing `path: nil`, or a persistent one by providing a file URL (e.g., `document.vdb`).

```swift
import VectorDatabase

// Create a persistent database storing 384-dimensional vectors (e.g., all-MiniLM-L6-v2)
let storeURL = URL(fileURLWithPath: "/path/to/storage")
let db = try VectorDatabase(dimension: 384, metric: .cosine, path: storeURL)
```

### 2. Insert Vectors
You can attach any key-value metadata to your vectors.
```swift
let embedding: [Float] = [0.12, 0.45, -0.89, /* ... 381 more ... */]

// Insert a single document
try await db.insert(id: "doc_1", vector: embedding, metadata: ["title": "Swift AI Guide", "author": "Apple"])

// Batch insert multiple documents for higher throughput
try await db.batchInsert([
    (id: "doc_2", vector: [...], metadata: nil),
    (id: "doc_3", vector: [...], metadata: ["tag": "swiftui"])
])
```

### 3. Search for Nearest Neighbors
Find the 5 most semantically similar vectors to your query.
```swift
let query: [Float] = [0.10, 0.40, -0.90, /* ... */]
let results = try await db.search(query: query, k: 5)

for result in results {
    print("Found \(result.id) with score \(result.score)")
    if let title = result.metadata?["title"] {
        print("Title: \(title)")
    }
}
```

### 4. Manage Data
Update metadata, fetch specific records, or delete them.
```swift
// Fetch a specific vector and its metadata
let record = await db.get(id: "doc_1")

// Update metadata instantly without re-inserting the large vector
try await db.updateMetadata(id: "doc_1", metadata: ["title": "Updated Swift Guide"])

// Delete a document
try await db.delete(id: "doc_1")
```

### 5. Persistence
Save your database to disk and close it properly.
```swift
// Manually save to disk (writes snapshot + truncates WAL)
try await db.save()

// Always close when done — this performs a final save
try await db.close()
```

---

## Distance Metrics

VectorDatabase supports three distance metrics. Choose the one that matches your embedding model:

| Metric | Best For | Score Convention |
|---|---|---|
| `.cosine` (default) | Text embeddings (OpenAI, Sentence-Transformers) | Higher = more similar. Vectors are auto-normalized at insert time. |
| `.dotProduct` | Pre-normalized embeddings, MaxIP tasks | Higher = more similar. Caller is responsible for normalization. |
| `.euclidean` | Spatial data, raw feature vectors | Higher = more similar (internally negated distance²). |

```swift
let db = try VectorDatabase(dimension: 384, metric: .euclidean, path: storeURL)
```

---

## HNSW Parameters Tuning

For advanced users, you can customize the HNSW index parameters:

```swift
let params = HNSWParameters(
    M: 16,              // Max edges per node (default: 16). Higher = better recall, more memory.
    efConstruction: 200, // Build-time candidate list (default: 200). Higher = better graph, slower insert.
    efSearch: 50,        // Query-time candidate list (default: 50). Higher = better recall, slower query.
    seed: 0x5EED_1234    // RNG seed for deterministic graph construction.
)
let db = try VectorDatabase(dimension: 384, metric: .cosine, path: storeURL, hnswParams: params)
```

| Parameter | Default | Effect |
|---|---|---|
| `M` | 16 | Number of bidirectional edges per node. Higher values improve recall at the cost of memory and insert speed. |
| `efConstruction` | 200 | Candidate list size during index build. Higher values produce a better graph but slow down inserts. |
| `efSearch` | 50 | Candidate list size at query time. The primary recall/latency knob. See [BENCHMARKS.md](BENCHMARKS.md) for sweep data. |
| `seed` | `0x5EED_1234` | Deterministic RNG seed for reproducible graph construction and tests. |

---

## Advanced: Batch Search

For high-throughput scenarios, use `searchBatch` to run multiple queries in parallel:

```swift
let queries: [[Float]] = [queryVec1, queryVec2, queryVec3]
let batchResults = try await db.searchBatch(queries: queries, k: 10)

for (i, results) in batchResults.enumerated() {
    print("Query \(i): top result = \(results.first?.id ?? "none")")
}
```

`searchBatch` automatically parallelizes across CPU cores using `DispatchQueue.concurrentPerform` for maximum throughput.

---

## Advanced: Graph Inspector

Want to visualize how the AI structures your data under the hood? `VectorDatabase` provides a read-only introspection API into its internal HNSW graph. This is perfect for building debugging tools or 3D graph visualizers.

```swift
let inspector = db.inspector

// Get the central entry point of the graph
if let entryPoint = await inspector.entryPointID() {
    
    // Check how many layers deep this node goes
    let layers = await inspector.layerCount(of: entryPoint)
    
    // Find exact topological neighbors at the base layer (Layer 0)
    let neighbors = await inspector.neighbors(of: entryPoint, atLayer: 0)
    print("Neighbors: \(neighbors ?? [])")
}
```

---

## Energy Profiling (Instruments)

Because VectorDatabase is built for iOS devices, **battery consumption** is just as critical as raw CPU speed.

Energy profiling lives in a companion iOS app rather than in this repository: **[HNSWEnergyProfiler](https://github.com/erenaskin/HNSWEnergyProfiler)**. It drives the `testEnergyProfileHNSWIOSDevice()` workload defined in [`Tests/VectorDatabaseTests/VectorDatabaseTests.swift`](Tests/VectorDatabaseTests/VectorDatabaseTests.swift) on a physical device so you can capture real energy numbers with Apple's **Instruments**.

### How to Profile Energy:
1. Clone and open the **[HNSWEnergyProfiler](https://github.com/erenaskin/HNSWEnergyProfiler)** app in Xcode.
2. Connect a physical iPhone/iPad (Energy profiling does not work on the Simulator).
3. Go to **Product > Profile** (⌘I) in Xcode.
4. In Instruments, choose the **Energy Log** template.
5. Run the app's 15-second workload, which exercises `VectorDatabase.insert`/`search` the same way `testEnergyProfileHNSWIOSDevice()` does.
6. Instruments will track the exact power overhead (CPU, overhead, and thermal state) of sustained inserts and searches.

You can also trigger the identical workload directly from this package's own test suite (skipped by default, since it runs for 15 seconds):
```bash
RUN_ENERGY_PROFILE=1 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test --filter VectorDatabaseTests.testEnergyProfileHNSWIOSDevice
```

---

## Documentation & Internals

| Document | Description |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | Detailed performance data, latency measurements, and recall sweeps |
| [TEST_GUIDE.md](TEST_GUIDE.md) | Comprehensive testing guide (terminal + Xcode), sanitizers, coverage |

---

## Roadmap

- **Concurrent Graph Construction**: `insert`/`batchInsert` currently run single-threaded under one exclusive write lock (`Engine` + `ReadWriteLock`). This is a deliberate design choice, not an oversight: `HNSWIndex.insert` rewires the neighbor lists of *other, already-existing* nodes as part of bidirectional linking and pruning, so a naive multi-threaded insert would race on shared adjacency lists.

  In practice this is not a bottleneck for the target use case (a personal, on-device dataset of tens of thousands to a few hundred thousand vectors, growing incrementally). It does become one for a one-time bulk load of 500k–1M+ vectors, where single-threaded insertion is CPU-bound and takes on the order of tens of minutes (see [BENCHMARKS.md](BENCHMARKS.md) for measured throughput).

  Making graph construction genuinely concurrent — in the spirit of hnswlib's per-node locking model — would require:
  1. Replacing the single global write lock with per-node locks (an insert only needs to lock the specific nodes it touches), plus a small dedicated lock/CAS for entry-point updates.
  2. Pre-sized or segmented storage for `VectorStorage`/`GraphStorage`: the current geometric `grow()` invalidates pointers and assumes exclusive access, which is incompatible with concurrent appends without either a fixed upfront capacity or a chunked allocator.
  3. Relaxing the current byte-identical determinism guarantee (seeded RNG + strict insertion order), since concurrent thread interleaving is not deterministic. Single-threaded mode would keep today's guarantee unchanged.

  This is a substantial, well-precedented architecture change (not a novel research problem) rather than a quick win, so it is tracked here as a future direction. A lower-risk interim step worth evaluating first: parallelizing the read-only candidate search (`searchLayer`) across threads while keeping the actual graph wiring serial, or sharding a large bulk load across N independent HNSW indices built in parallel and merged behind a query-time router.

---

## Known Limitations

- **Apple-only**: Relies on Apple's `Accelerate` framework (vDSP, BLAS) and Darwin APIs (`mmap`, `flock`, `os_unfair_lock`). Not portable to Linux.
- **No multi-process mutation**: Cross-process safety covers `save()`/`load()` only. Concurrent mutation from multiple processes between saves is not supported.
- **No vector updates**: To change a vector, delete and re-insert. Metadata can be updated in-place.
- **Int32 ID space**: Internal IDs are `Int32`, capping at ~2.1 billion vectors per database.

---

## Semantic Versioning Policy

`VectorDatabase` adheres strictly to Semantic Versioning (SemVer), particularly regarding data persistence to guarantee your app updates never break user data.

- **Major versions (`x.0.0`)**: Indicate a breaking change to the **on-disk binary format**. 
  - The on-disk header contains a `formatVersion`. When the SDK updates to a new major version, it may not be able to silently read older files without a manual migration step. This guarantees that SDK updates will **never silently corrupt** your users' local disk stores. 
- **Minor/Patch versions (`1.x.x`)**: Are guaranteed to be fully backwards-compatible at the storage layer. (e.g. WAL format expansions).
