# SwiftVectorDB — Development Guide
### An Embedded, Pure-Swift Vector Database for On-Device Semantic Search & RAG

**Status:** Pre-implementation reference document
**Target platforms:** iOS 15+ / macOS 12+ (Accelerate + Swift Concurrency requirements)
**Language:** Swift only — no C++ dependencies, no external native libraries

---

## How to Use This Document

This is meant to be your single source of truth from empty folder to shipped package. It is ordered as a **build sequence** — each phase produces a working, testable increment. Do not skip Phase 1 to jump to HNSW; the brute-force flat index you build there becomes your permanent correctness oracle for everything after it.

Each phase has: **Goal → Concepts → Design Decisions → Implementation → Code Skeleton → Pitfalls → Definition of Done**.

---

## Table of Contents

1. [Vision & Non-Goals](#1-vision--non-goals)
2. [Environment & Project Setup](#2-environment--project-setup)
3. [Project Structure](#3-project-structure)
4. [Architecture Overview](#4-architecture-overview)
5. [Phase 1 — Flat Index & Brute-Force Search (MVP)](#5-phase-1--flat-index--brute-force-search-mvp)
6. [Phase 2 — SIMD Acceleration with Accelerate](#6-phase-2--simd-acceleration-with-accelerate)
7. [Phase 3 — Unsafe Memory & Contiguous Storage](#7-phase-3--unsafe-memory--contiguous-storage)
8. [Phase 4 — HNSW Graph Index](#8-phase-4--hnsw-graph-index)
9. [Phase 5 — Concurrency & Thread Safety](#9-phase-5--concurrency--thread-safety)
10. [Phase 6 — Persistence: File Format & Memory-Mapped I/O](#10-phase-6--persistence-file-format--memory-mapped-io)
11. [Phase 7 — Deletion & Updates](#11-phase-7--deletion--updates)
12. [Phase 8 — Public API Design](#12-phase-8--public-api-design)
13. [Phase 9 — Testing Strategy](#13-phase-9--testing-strategy)
14. [Phase 10 — Benchmarking & Tuning](#14-phase-10--benchmarking--tuning)
15. [Phase 11 — Packaging, CI, Distribution](#15-phase-11--packaging-ci-distribution)
16. [Edge Case & Failure Mode Checklist](#16-edge-case--failure-mode-checklist)
17. [Glossary](#17-glossary)
18. [References](#18-references)
19. [Milestone Checklist](#19-milestone-checklist)

---

## 1. Vision & Non-Goals

**Vision:** A zero-dependency Swift Package that lets an iOS developer do this in five lines:

```swift
let db = try VectorDB(dimension: 384, metric: .cosine, path: storeURL)
try await db.insert(id: "note_1", vector: embedding)
let results = try await db.search(query: queryEmbedding, k: 5)
```

Everything else — graph construction, SIMD math, disk paging — is invisible to the caller.

**Non-goals (explicitly out of scope, at least for v1):**
- Distributed/multi-node operation (this is embedded, single-process)
- Arbitrary metadata query language (filtering is a v2 feature; v1 only stores an opaque `[String: String]` metadata bag per vector, returned as-is with search results — no querying/indexing on it)
- Non-Apple platforms (Linux Accelerate does not exist; you'd need a `vDSP`-less fallback, deprioritize)
- GPU (Metal) compute — mentioned as a future optimization path, not required for v1. Accelerate/vDSP on CPU is already extremely fast for the vector dimensions typical of on-device embeddings (256–1536) and avoids GPU dispatch overhead for small batches.
- Exact (non-approximate) search at scale — HNSW is approximate by design; you will always trade a small amount of recall for speed.

---

## 2. Environment & Project Setup

### Requirements
- Xcode 15+ (Swift 5.9+, for `~Copyable`, macros if you want them, and modern concurrency)
- A physical iOS device for battery/perf testing eventually — the simulator does not reflect real Accelerate/CPU behavior accurately, especially for thermal throttling.

### Bootstrap the package

```bash
mkdir SwiftVectorDB && cd SwiftVectorDB
swift package init --type library --name VectorDB
```

### `Package.swift` starting point

```swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "VectorDB",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(name: "VectorDB", targets: ["VectorDB"]),
    ],
    dependencies: [
        // Optional — see Phase 4 discussion on priority queues.
        // .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "VectorDB",
            dependencies: [/* "Collections", .product(name: "Collections", package: "swift-collections") if used */],
            swiftSettings: [
                .unsafeFlags(["-Ounchecked"], .when(configuration: .release))
            ]
        ),
        .testTarget(name: "VectorDBTests", dependencies: ["VectorDB"]),
    ]
)
```

> **Design decision — dependency policy:** The prompt calls for "pure-Swift." `swift-collections` (Apple's own package) is pure Swift with no C interop, so it does not violate that spirit and gives you a battle-tested `Heap`. But if you want *zero* dependencies for maximum portability and minimal supply-chain risk, implement your own binary heap (~80 lines, given in Phase 4). This guide gives you both paths; pick one early and don't mix them.

> **`-Ounchecked` note:** This disables array bounds/overflow checks in release builds, which matters a lot once you're doing millions of float operations. Do **not** enable it until Phase 1's correctness tests are green — you want crashes, not silent corruption, while you're still debugging logic.

---

## 3. Project Structure

```
SwiftVectorDB/
├── Package.swift
├── Sources/
│   └── VectorDB/
│       ├── Public/
│       │   ├── VectorDB.swift            // public actor — main entry point
│       │   ├── VectorDBError.swift
│       │   ├── DistanceMetric.swift
│       │   ├── SearchResult.swift
│       │   └── HNSWParameters.swift
│       ├── Core/
│       │   ├── FlatIndex.swift           // Phase 1 brute-force engine
│       │   ├── HNSWIndex.swift           // Phase 4 graph engine
│       │   └── IndexProtocol.swift       // shared interface both conform to
│       ├── Storage/
│       │   ├── VectorStorage.swift       // unsafe contiguous float storage
│       │   ├── GraphStorage.swift        // adjacency list storage
│       │   ├── MappedFile.swift          // mmap wrapper
│       │   ├── FileFormat.swift          // header/layout constants
│       │   ├── WriteAheadLog.swift
│       │   └── PersistenceManager.swift  // save/load/compaction orchestration
│       ├── Math/
│       │   └── VectorMath.swift          // vDSP-backed distance functions
│       ├── Concurrency/
│       │   └── ReadWriteLock.swift
│       └── Utilities/
│           ├── BinaryHeap.swift          // only if not using swift-collections
│           └── IDMap.swift               // String id <-> Int32 internal index + optional per-vector metadata dictionary (see §10.2, §12)
├── Tests/
│   └── VectorDBTests/
│       ├── FlatIndexTests.swift
│       ├── HNSWCorrectnessTests.swift    // recall@k vs. FlatIndex oracle
│       ├── PersistenceTests.swift
│       ├── ConcurrencyTests.swift
│       ├── EdgeCaseTests.swift
│       └── TestFixtures.swift            // deterministic random vector generators
└── Benchmarks/
    └── VectorDBBenchmarks/
        └── main.swift                    // standalone executable, not XCTest
```

**Why separate `FlatIndex` from `HNSWIndex` permanently (not just as a throwaway prototype):** you will use `FlatIndex` forever as (a) the ground-truth oracle in tests, (b) the fallback for tiny datasets where building a graph isn't worth it, and (c) the re-ranking step after an HNSW candidate retrieval (a very common production pattern: get top-200 approximate candidates fast, then brute-force re-rank the top-200 exactly).

**Why `IndexProtocol`:** so `VectorDB` (the public actor) can hold either implementation behind one interface and switch between them based on collection size, without the public API changing.

```swift
protocol VectorIndex {
    var dimension: Int { get }
    var count: Int { get }
    mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws
    mutating func remove(internalID: Int32) throws
    func search(query: UnsafePointer<Float>, k: Int, ef: Int?) -> [(id: Int32, score: Float)]
}
```

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Public API (actor VectorDB)                                 │
│  insert() / search() / delete() / save() / close()           │
└───────────────┬───────────────────────────────────────────────┘
                │  String id <-> Int32 internal id (IDMap)
┌───────────────▼───────────────────────────────────────────────┐
│  Core Index (FlatIndex  or  HNSWIndex)                        │
│   - owns the graph / list logic                                │
│   - calls into VectorMath for all distance calcs               │
└───────────────┬─────────────────────────┬──────────────────────┘
                │                          │
┌───────────────▼───────────────┐ ┌────────▼────────────────────┐
│  VectorStorage                 │ │  GraphStorage                │
│  contiguous UnsafeMutablePointer│ │  fixed-width adjacency lists │
│  <Float>, row-major, N x D      │ │  per node per layer          │
└───────────────┬───────────────┘ └────────┬────────────────────┘
                │                          │
┌───────────────▼──────────────────────────▼────────────────────┐
│  PersistenceManager                                             │
│   - MappedFile (mmap)  +  WriteAheadLog  +  snapshot/compaction  │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow for `insert`:**
1. Public API validates dimension, converts `[Float]` → written into `VectorStorage` at a new internal slot.
2. `IDMap` records `String id → Int32 slot`.
3. Core index (HNSW) runs its insertion algorithm, writing neighbor lists into `GraphStorage`.
4. `PersistenceManager` appends an insert record to the WAL (fast, durable) — the mmap'd snapshot file is *not* touched synchronously.
5. Periodically (or on `save()`), WAL is compacted into a fresh snapshot.

**Data flow for `search`:**
1. Public API validates dimension, copies query into a temporary aligned buffer.
2. Core index runs greedy graph descent (HNSW) using `VectorMath` distance calls against `VectorStorage`.
3. Results (internal IDs + scores) are mapped back to `String` ids + metadata via `IDMap`.

---

## 5. Phase 1 — Flat Index & Brute-Force Search (MVP)

### Goal
A working, correct, *slow* vector store: `O(N)` search, plain `[Float]` storage, no Accelerate, no unsafe pointers yet. This is intentionally the "boring" version — its entire purpose is to exist as ground truth later.

### Design Decisions
- Store vectors as `[[Float]]` or better, `[Float]` flattened with manual stride math (`vectors[i * dim ..< i * dim + dim]`) — get used to row-major layout now since it carries through the whole project.
- Distance metric abstraction from day one — don't hardcode cosine.
- `VectorMath` is introduced *here*, in Phase 1, as a plain scalar-loop implementation (no Accelerate, no unsafe pointers) so the phase is genuinely self-contained. Phase 2 keeps the exact same function signatures and swaps the bodies for `vDSP`/BLAS calls — the rest of the codebase never needs to change when that swap happens.

### Implementation

```swift
public enum DistanceMetric: Sendable {
    case cosine
    case euclidean
    case dotProduct
}

/// Naive scalar version — correct but slow. Phase 2 replaces the internals
/// with vDSP/BLAS calls while keeping these exact signatures.
enum VectorMath {
    static func similarity(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
                            _ dim: Int, metric: DistanceMetric) -> Float {
        switch metric {
        case .dotProduct, .cosine:
            var sum: Float = 0
            for i in 0..<dim { sum += a[i] * b[i] }
            return sum   // for .cosine, caller is expected to have normalized vectors at insert time
        case .euclidean:
            var distSq: Float = 0
            for i in 0..<dim { let d = a[i] - b[i]; distSq += d * d }
            return -distSq
        }
    }
}

struct FlatIndex: VectorIndex {
    let dimension: Int
    let metric: DistanceMetric
    private(set) var count: Int = 0
    private var flatVectors: [Float] = []   // row-major, count * dimension
    private var idToSlot: [Int32: Int] = [:]
    private var slotToID: [Int32] = []

    mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
        flatVectors.append(contentsOf: UnsafeBufferPointer(start: vector, count: dimension))
        idToSlot[internalID] = count
        slotToID.append(internalID)
        count += 1
    }

    func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)] {
        var scored: [(Int32, Float)] = []
        scored.reserveCapacity(count)
        flatVectors.withUnsafeBufferPointer { buf in
            for i in 0..<count {
                let base = buf.baseAddress! + i * dimension
                let score = VectorMath.similarity(query, base, dimension, metric: metric)
                scored.append((slotToID[i], score))
            }
        }
        scored.sort { $0.1 > $1.1 }  // higher = more similar, for cosine/dot; invert for euclidean distance
        return Array(scored.prefix(k))
    }

    mutating func remove(internalID: Int32) throws {
        // Swap-remove: O(1) but reorders — must update idToSlot/slotToID accordingly.
        // Implement once VectorStorage exists properly (Phase 3); for the MVP a naive
        // "mark and filter" is fine.
    }
}
```

### Pitfalls
- **Sort direction bugs**: cosine/dot-product similarity is "higher is better," Euclidean *distance* is "lower is better." Standardize internally: always convert to a "score, higher = better" convention at the `VectorMath` layer so the rest of the code never branches on metric type. E.g., for Euclidean, return `-distance` or `1/(1+distance)` as the score.
- **Empty index search**: `search` on `count == 0` must return `[]`, not crash on `buf.baseAddress!`.
- **`k > count`**: clamp `k` to `count`, don't throw.

### Definition of Done
- [ ] Insert 10k random vectors, search returns correct top-k by manual verification against a Python/numpy reference (`numpy.dot` + `argsort`) for at least 5 queries.
- [ ] Empty-index and `k=0` edge cases handled without crashing.
- [ ] Unit tests pass for all three metrics.

---

## 6. Phase 2 — SIMD Acceleration with Accelerate

### Goal
Replace the scalar Swift loop inside the `VectorMath` enum from Phase 1 with `vDSP`/BLAS calls, keeping the same `similarity(...)` signature so `FlatIndex` and everything built on top of it needs zero changes. This alone typically gives a 5–20x speedup on a single similarity calculation, and far more when batched.

### Concepts
- **vDSP** operates on `UnsafePointer<Float>` buffers with a stride — no Swift `Array` overhead once you're inside the call.
- Cosine similarity = `dot(a,b) / (||a|| * ||b||)`. If you **normalize vectors to unit length at insert time**, cosine similarity degenerates to a plain dot product at query time — this is the single biggest practical speedup in this whole project and costs almost nothing (normalize once, not per query). Do this.
- For batches (comparing one query against many stored vectors at once), `cblas_sgemm` (matrix × matrix) is dramatically faster than looping `vDSP_dotpr` once per vector, because BLAS is cache-blocked and takes advantage of the CPU's full SIMD width across the whole batch. Use this for the Phase 1 flat re-ranking step and for the layer-0 brute-force fallback for tiny collections.

### Implementation

```swift
import Accelerate

enum VectorMath {

    /// Score convention: higher = more similar, for ALL metrics.
    static func similarity(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
                            _ dim: Int, metric: DistanceMetric) -> Float {
        switch metric {
        case .dotProduct:
            return dot(a, b, dim)
        case .cosine:
            // Assumes vectors are pre-normalized at insert time (see normalize()).
            // If you cannot guarantee that, fall back to the unnormalized path below.
            return dot(a, b, dim)
        case .euclidean:
            var distSq: Float = 0
            vDSP_distancesq(a, 1, b, 1, &distSq, vDSP_Length(dim))
            return -distSq   // negate: lower distance -> higher score
        }
    }

    static func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ dim: Int) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(dim))
        return result
    }

    /// In-place L2 normalization. Call once at insert time for cosine-metric collections.
    static func normalize(_ v: UnsafeMutablePointer<Float>, _ dim: Int) {
        var sumSq: Float = 0
        vDSP_svesq(v, 1, &sumSq, vDSP_Length(dim))
        guard sumSq > 1e-12 else { return }   // guard against zero vectors — see pitfalls
        var norm = sqrtf(sumSq)
        var scale = 1.0 / norm
        vDSP_vsmul(v, 1, &scale, v, 1, vDSP_Length(dim))
    }

    /// Batched dot product: one query against N stored vectors, using BLAS.
    /// `vectors` is row-major N x dim. Returns N scores.
    static func batchDot(query: UnsafePointer<Float>,
                          vectors: UnsafePointer<Float>,
                          count: Int, dim: Int) -> [Float] {
        var results = [Float](repeating: 0, count: count)
        // y = M x  ->  using cblas_sgemv: matrix (count x dim) times vector (dim) = vector (count)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    Int32(count), Int32(dim),
                    1.0, vectors, Int32(dim),
                    query, 1,
                    0.0, &results, 1)
        return results
    }
}
```

> **Why `sgemv` not `sgemm` here:** one query against N vectors is a matrix-vector product (`sgemv`), which is what you'll use during HNSW's flat re-ranking or the small-collection fallback. Reach for `sgemm` (matrix-matrix) only when you have **multiple queries at once** — e.g., batch-embedding a whole document and searching all chunks together. Don't over-engineer the single-query hot path with `sgemm`.

### Pitfalls
- **Zero vectors**: normalizing a zero vector divides by zero → `NaN` propagates silently through everything downstream and is very hard to trace back. Guard explicitly (as above) and consider throwing `VectorDBError.invalidVector` on insert if `sumSq` is below a tiny epsilon, rather than silently skipping normalization.
- **NaN/Inf inputs**: CoreML models can occasionally emit `NaN` on malformed input. Validate on insert with `vDSP_vsanity`-style manual checks, or just scan with `v.contains(where: { !$0.isFinite })` before accepting the vector — cheap relative to the cost of a corrupted graph.
- **Memory alignment**: `vDSP` doesn't strictly require 16/32-byte alignment on Apple Silicon the way older SSE code did, but allocating with `UnsafeMutablePointer<Float>.allocate(capacity:)` already gives you suitably aligned memory — don't hand-roll alignment logic.
- **Dimension mismatch**: always validate `vector.count == dimension` in the public API layer *before* it reaches any unsafe pointer code. An off-by-N dimension here is a buffer overrun, not a Swift-safe out-of-bounds trap, once you're in Phase 3's raw pointer world.

### Definition of Done
- [ ] Benchmark: vDSP dot product vs. naive `zip(a,b).reduce` loop, on 384-dim vectors, 100k iterations — confirm meaningful speedup (expect 5-15x on device).
- [ ] Cosine similarity via pre-normalization gives identical results (within float epsilon) to the textbook `dot/(normA*normB)` formula.
- [ ] Zero-vector insert is rejected or safely handled, verified by a unit test.

---

## 7. Phase 3 — Unsafe Memory & Contiguous Storage

### Goal
Replace `[Float]` (Phase 1) with a manually-managed contiguous buffer that can grow, be memory-mapped later, and be indexed with zero abstraction overhead.

### Design Decisions
- One giant `UnsafeMutablePointer<Float>` for **all** vectors, row-major: vector `i` lives at `base + i * dimension`.
- Growth strategy: geometric doubling (like `Array` does internally), because reallocating on every single insert is O(N²) total.
- Wrap the raw pointer in a `final class` (not a struct) so you can implement `deinit` to free memory — Swift won't do this for you with unsafe pointers.

### Implementation

```swift
final class VectorStorage {
    let dimension: Int
    private(set) var capacity: Int
    private(set) var count: Int = 0
    private var buffer: UnsafeMutablePointer<Float>

    init(dimension: Int, initialCapacity: Int = 1024) {
        self.dimension = dimension
        self.capacity = initialCapacity
        self.buffer = UnsafeMutablePointer<Float>.allocate(capacity: initialCapacity * dimension)
    }

    deinit {
        buffer.deallocate()
    }

    /// Returns the slot index the vector was written to.
    @discardableResult
    func append(_ vector: UnsafePointer<Float>) -> Int {
        if count == capacity { grow() }
        let dest = buffer + count * dimension
        dest.update(from: vector, count: dimension)
        let slot = count
        count += 1
        return slot
    }

    func pointer(toSlot slot: Int) -> UnsafePointer<Float> {
        precondition(slot >= 0 && slot < count, "VectorStorage: slot \(slot) out of bounds")
        return UnsafePointer(buffer + slot * dimension)
    }

    func mutablePointer(toSlot slot: Int) -> UnsafeMutablePointer<Float> {
        precondition(slot >= 0 && slot < count, "VectorStorage: slot \(slot) out of bounds")
        return buffer + slot * dimension
    }

    private func grow() {
        let newCapacity = capacity * 2
        let newBuffer = UnsafeMutablePointer<Float>.allocate(capacity: newCapacity * dimension)
        newBuffer.update(from: buffer, count: count * dimension)
        buffer.deallocate()
        buffer = newBuffer
        capacity = newCapacity
    }
}
```

### Pitfalls — read this section twice
- **Use-after-free via `pointer(toSlot:)`**: if you hold a pointer returned by `pointer(toSlot:)` across a call that might trigger `grow()` (i.e., any subsequent `append`), that pointer is now dangling — `grow()` deallocates the old buffer. **Rule: never cache a raw pointer from `VectorStorage` across an insert call.** Re-fetch it after any mutation. This is the single most dangerous bug class in this whole project; consider adding a debug-only generation counter that increments on `grow()` and have `pointer(toSlot:)` callers assert against a captured generation if you want to catch this in tests.
- **`precondition` vs. silent corruption**: bounds-check in debug/testing builds with `precondition`. Once you flip on `-Ounchecked` for release, these checks may be compiled out for related `Array`/`UnsafeBufferPointer` subscripting — but `precondition` itself (unlike `assert`) still traps in release by default unless you specifically use `-Ounchecked`'s effect on it; confirm this via a crash test before relying on it. If unsure, keep manual bounds checks (`if slot >= count { fatalError(...) }`) rather than trusting compiler flag semantics you haven't verified.
- **Copy semantics**: `VectorStorage` is a `class` for a reason — if it were a `struct`, Swift's value semantics would create *shallow* copies for any raw pointer field (the pointer would be copied, not the underlying memory), and you'd get double-free crashes when both copies deinit. Never make this a `struct` unless you also implement full copy-on-write, which is significant extra complexity you don't need for an internal storage engine.
- **Deallocating memory still referenced elsewhere**: if `HNSWNode` or any other structure stores a raw `UnsafePointer<Float>` into this buffer, and you later resize the buffer or deallocate `VectorStorage` itself, those become dangling. Prefer storing **slot indices** (`Int32`) everywhere outside `VectorStorage`, and only materialize a pointer at the point of use, immediately before a math call. Do not persist raw pointers in long-lived structures.
- **Thread safety is not handled here at all** — `VectorStorage` as written above assumes single-threaded access. See Phase 5 for the locking strategy that wraps this.

### Definition of Done
- [ ] Insert 1M vectors of dimension 384 without a leak (verify with Instruments' Leaks/Allocations tool).
- [ ] Stress test: interleave `append` and search reads on a single thread, confirm growth doesn't corrupt existing data (compare retrieved vectors byte-for-byte against a reference `[[Float]]` array kept in the test for verification).
- [ ] Explicit unit test that would have caught a "dangling pointer across grow()" bug, e.g. capture a pointer, force 10 more inserts to guarantee a resize, then verify a fresh `pointer(toSlot:)` call returns correct (not garbage) data — this indirectly proves you're not relying on stale pointers anywhere in calling code.

---

## 8. Phase 4 — HNSW Graph Index

This is the algorithmic core of the project. Read the original paper (Malkov & Yashunin — see §18 References for the full citation) alongside this section; this guide gives you an implementation-focused translation, not a substitute for understanding the theory.

### 8.1 Core Concepts

HNSW builds several layers of a proximity graph, like a skip list generalized to graphs:
- **Layer 0** contains *every* inserted vector, densely connected to its nearest neighbors.
- **Higher layers** contain exponentially fewer vectors (each vector has a random chance of "promoting" to the next layer up), acting as express lanes for long-distance jumps during search.
- Search starts at the top (sparse) layer, greedily walks toward the query, drops down a layer once it can't improve further, and repeats until layer 0, where it does a wider beam search for the final candidate set.

### 8.2 Parameters — get these right, they dominate the recall/speed/memory tradeoff

| Parameter | Meaning | Typical value | Effect |
|---|---|---|---|
| `M` | Max neighbors per node, per layer (layers ≥ 1) | 12–48 | Higher = better recall, more memory, slower insert |
| `Mmax0` | Max neighbors per node at layer 0 | `2 * M` | Layer 0 needs denser connectivity since it's the final search layer |
| `efConstruction` | Candidate list size during insertion | 100–200 | Higher = better graph quality, much slower build |
| `efSearch` | Candidate list size during query | ≥ `k`, typically 50–200 | Higher = better recall, slower query. This is your **runtime-tunable knob** — expose it in the public API per-query |
| `mL` | Level-generation normalization factor | `1 / ln(M)` | Controls how many nodes end up on higher layers |

```swift
public struct HNSWParameters: Sendable {
    public var M: Int = 16
    public var Mmax0: Int
    public var efConstruction: Int = 200
    public var efSearch: Int = 64
    public var mL: Double
    public var seed: UInt64 = 0x5EED

    public init(M: Int = 16, efConstruction: Int = 200, efSearch: Int = 64, seed: UInt64 = 0x5EED) {
        self.M = M
        self.Mmax0 = M * 2
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.mL = 1.0 / log(Double(M))
        self.seed = seed
    }

    public static let `default` = HNSWParameters()
}
```

> **On-device recommendation:** start with `M=16, efConstruction=100, efSearch=64` for note/text-embedding use cases (hundreds to low tens of thousands of vectors). These favor build/insert speed over the marginal recall gains you'd chase at web-scale. Tune upward only if your recall benchmarks (Phase 9) show a real problem.

### 8.3 Data Structures

```swift
struct HNSWNode {
    var level: Int                     // top layer this node exists on
    var vectorSlot: Int32               // index into VectorStorage
}
```

Neighbor lists are **not** stored as `[[Int32]]` per node (too much pointer-chasing and allocation overhead for something you touch millions of times). Instead, use a fixed-width scheme in `GraphStorage`:

```swift
/// Fixed-capacity adjacency storage: every node reserves Mmax0 slots at layer 0
/// and M slots at each higher layer it participates in, padded with `emptySlot`.
/// This gives O(1) offset math — critical for both cache performance and later mmap addressing.
final class GraphStorage {
    static let emptySlot: Int32 = -1

    let mMax0: Int
    let m: Int
    private var layer0: UnsafeMutablePointer<Int32>       // count * mMax0
    private var upperLayers: [UnsafeMutablePointer<Int32>] // per level: count * m (only for nodes that reach that level)
    private(set) var capacity: Int
    private(set) var count: Int = 0

    // Neighbor count is tracked separately since slots are padded with emptySlot,
    // but a fast "how many neighbors does node X have at layer L" is needed often
    // enough during insert that caching it avoids a full row scan.
    private var neighborCounts: [[Int32: Int]] = []   // [layer][internalID] -> count, sparse for upper layers

    init(m: Int, mMax0: Int, initialCapacity: Int = 1024) {
        self.m = m
        self.mMax0 = mMax0
        self.capacity = initialCapacity
        self.layer0 = UnsafeMutablePointer<Int32>.allocate(capacity: initialCapacity * mMax0)
        self.layer0.initialize(repeating: Self.emptySlot, count: initialCapacity * mMax0)
    }

    deinit {
        layer0.deallocate()
        upperLayers.forEach { $0.deallocate() }
    }

    // neighbors(of:at:), setNeighbors(of:at:to:), and the growth/level-allocation
    // logic follow the same allocate-copy-deallocate pattern as VectorStorage.
    // Implement per-layer growth lazily: only allocate an upperLayers[level] buffer
    // the first time a node actually reaches that level.
}
```

> **Design tradeoff you must consciously make:** fixed-width adjacency (above) wastes some memory on partially-filled neighbor lists but gives you O(1) addressing, which is essential once this data lives in an mmap'd file (Phase 6) where you cannot afford to parse variable-length records to find an offset. A variable-length / `[[Int32]]`-based adjacency list is simpler to write today but you will have to redesign it entirely for persistence. **Build the fixed-width version from the start.**

### 8.4 Level Assignment

```swift
extension HNSWIndex {
    /// Malkov & Yashunin's exponential decay level assignment.
    func randomLevel() -> Int {
        let r = Double.random(in: 0..<1, using: &rng)
        return Int(floor(-log(r) * params.mL))
    }
}
```

Use a **seeded PRNG** (e.g. a simple splitmix64/xorshift you control, or `SystemRandomNumberGenerator` wrapped with a fixed seed for tests) rather than the default global `Double.random`. You need deterministic graph construction for reproducible tests — "does my recall regression test still pass" is meaningless if the graph shape changes randomly between runs.

### 8.5 Search Layer (the core primitive, used by both insert and query)

This is a direct translation of the paper's `SEARCH-LAYER` algorithm.

```swift
/// Returns up to `ef` nearest candidates to `query` found by greedy graph traversal
/// within a single layer, starting from `entryPoints`.
private func searchLayer(query: UnsafePointer<Float>,
                          entryPoints: [Int32],
                          ef: Int,
                          layer: Int) -> [Candidate] {
    var visited = Set<Int32>(entryPoints)
    // candidates: min-heap by distance (closest first) — nodes still to explore
    var candidates = MinHeap<Candidate>(entryPoints.map { scored($0, query) })
    // found: max-heap by distance (farthest first) — current best-`ef` result set
    var found = MaxHeap<Candidate>(candidates.elements)

    while let c = candidates.popMin() {
        guard let worst = found.peekMax() else { break }
        if c.score < worst.score && found.count >= ef {
            break // nothing left in the candidate queue can beat our current worst
        }
        for neighborID in graphStorage.neighbors(of: c.id, at: layer) {
            guard !visited.contains(neighborID) else { continue }
            visited.insert(neighborID)
            let candidateScore = scored(neighborID, query)
            if found.count < ef || candidateScore.score > found.peekMax()!.score {
                candidates.push(candidateScore)
                found.push(candidateScore)
                if found.count > ef { found.popMax() }
            }
        }
    }
    return found.sortedDescending()
}
```

> **Note on the heap comparator direction**: this guide's `VectorMath` convention is "higher score = more similar," which is the *opposite* sense of "distance" in the original paper's pseudocode (which minimizes distance). Keep this consistent everywhere — a classic and painful bug class in HNSW ports is silently mixing distance-minimization and similarity-maximization comparators between the two heaps. Write a unit test that directly checks heap ordering with known values before trusting anything built on top of it.

### 8.6 Insert

```swift
mutating func insert(internalID: Int32, vector: UnsafePointer<Float>) throws {
    let slot = vectorStorage.append(vector)
    let level = randomLevel()

    guard let entryID = entryPoint else {
        // First node in the index — becomes entry point at its own level, no edges.
        entryPoint = internalID
        entryPointLevel = level
        nodes[internalID] = HNSWNode(level: level, vectorSlot: Int32(slot))
        graphStorage.allocateNode(internalID, upToLevel: level)
        return
    }

    var currentNearest = [entryID]
    // Phase A: descend from top layer to just above the new node's level,
    // doing greedy ef=1 search to find a good entry point at each layer.
    if level < entryPointLevel {
        for lc in stride(from: entryPointLevel, to: level, by: -1) {
            currentNearest = searchLayer(query: vector, entryPoints: currentNearest, ef: 1, layer: lc).map(\.id)
        }
    }

    graphStorage.allocateNode(internalID, upToLevel: level)

    // Phase B: from min(entryPointLevel, level) down to 0, do full efConstruction search,
    // pick neighbors, wire bidirectional edges, and prune any neighbor that overflows Mmax.
    for lc in stride(from: min(entryPointLevel, level), through: 0, by: -1) {
        let candidates = searchLayer(query: vector, entryPoints: currentNearest, ef: params.efConstruction, layer: lc)
        let maxConn = (lc == 0) ? params.Mmax0 : params.M
        let selected = selectNeighborsHeuristic(candidates: candidates, m: maxConn)

        graphStorage.setNeighbors(of: internalID, at: lc, to: selected.map(\.id))
        for neighbor in selected {
            graphStorage.addNeighbor(of: neighbor.id, at: lc, neighborID: internalID)
            let neighborMax = (lc == 0) ? params.Mmax0 : params.M
            if graphStorage.neighborCount(of: neighbor.id, at: lc) > neighborMax {
                let prunedCandidates = graphStorage.neighbors(of: neighbor.id, at: lc)
                    .map { scored($0, vectorStorage.pointer(toSlot: Int(nodes[neighbor.id]!.vectorSlot))) }
                let pruned = selectNeighborsHeuristic(candidates: prunedCandidates, m: neighborMax)
                graphStorage.setNeighbors(of: neighbor.id, at: lc, to: pruned.map(\.id))
            }
        }
        currentNearest = candidates.map(\.id)
    }

    if level > entryPointLevel {
        entryPoint = internalID
        entryPointLevel = level
    }
    nodes[internalID] = HNSWNode(level: level, vectorSlot: Int32(slot))
}
```

**Neighbor selection heuristic**: the naive approach (just take the `m` closest candidates) tends to create clustered, poorly-connected graphs. The paper's heuristic favors diversity — prefer a candidate only if it's closer to the query than it is to any *already-selected* neighbor. Implement this (it's maybe 20 lines) rather than the naive top-m; recall improves meaningfully for a small extra insert cost:

```swift
private func selectNeighborsHeuristic(candidates: [Candidate], m: Int) -> [Candidate] {
    var sorted = candidates.sorted { $0.score > $1.score }  // closest (highest score) first
    var selected: [Candidate] = []
    while let candidate = sorted.first, selected.count < m {
        sorted.removeFirst()
        let candidateVec = vectorStorage.pointer(toSlot: Int(nodes[candidate.id]!.vectorSlot))
        let isDiverse = selected.allSatisfy { existing in
            let existingVec = vectorStorage.pointer(toSlot: Int(nodes[existing.id]!.vectorSlot))
            let distToExisting = VectorMath.similarity(candidateVec, existingVec, dimension, metric: metric)
            return candidate.score > distToExisting
        }
        if isDiverse || selected.isEmpty {
            selected.append(candidate)
        }
    }
    // If diversity pruning left us short of m, backfill with the next-closest remaining candidates.
    if selected.count < m {
        selected.append(contentsOf: sorted.prefix(m - selected.count))
    }
    return selected
}
```

### 8.7 Query (K-NN Search)

```swift
func search(query: UnsafePointer<Float>, k: Int, ef: Int? = nil) -> [(id: Int32, score: Float)] {
    guard let entryID = entryPoint else { return [] }
    let efSearch = max(ef ?? params.efSearch, k)

    var currentNearest = [entryID]
    for lc in stride(from: entryPointLevel, to: 0, by: -1) {
        currentNearest = searchLayer(query: query, entryPoints: currentNearest, ef: 1, layer: lc).map(\.id)
    }
    let results = searchLayer(query: query, entryPoints: currentNearest, ef: efSearch, layer: 0)
    return results.prefix(k).map { ($0.id, $0.score) }
}
```

### Pitfalls
- **Disconnected graph after many deletions** — see Phase 7; HNSW was not designed for deletion, plan for it explicitly rather than bolting it on later.
- **Small collections**: HNSW has real overhead (graph construction, multiple heap allocations) that isn't worth it below a few thousand vectors. Have `VectorDB` transparently use `FlatIndex` below a configurable threshold (e.g. 2,000 vectors) and switch to `HNSWIndex` above it — same public API, invisible to the caller. This directly serves the "personal notes app" use case where a given user might genuinely have only 300 notes.
- **Non-determinism in tests**: seed your RNG (§8.4) or every test run produces a structurally different graph, making recall regressions non-reproducible.
- **`entryPointLevel` staying stale after deletions** — if the current entry point gets deleted, you must pick a new one (any remaining node works, but prefer one from the highest surviving layer) or every subsequent insert/search silently operates on a dangling reference.
- **Recursive lock reentrancy**: `selectNeighborsHeuristic`'s pruning step, called from within `insert`, mutates neighbor lists of *other* nodes while you're still "inside" the top-level insert call. If you've wrapped mutation with a lock (Phase 5), make sure this doesn't attempt to reacquire a non-reentrant lock.

### Definition of Done
- [ ] Recall@10 ≥ 0.95 against the `FlatIndex` oracle on a 50k-vector synthetic dataset, at your chosen default `efSearch`.
- [ ] Insert 50k vectors completes in a reasonable wall-clock time on a real device (benchmark and record the number — this becomes your regression baseline).
- [ ] Deterministic graph construction verified: same seed + same insert order ⇒ byte-identical graph structure across two runs.
- [ ] Heap comparator correctness unit-tested independently of the rest of HNSW.

---

## 9. Phase 5 — Concurrency & Thread Safety

### Design Decisions
iOS apps call into SDKs from arbitrary queues (main thread UI triggers, background indexing tasks, etc). You need to support **concurrent reads** (many simultaneous searches) and **exclusive writes** (insert/delete), which is a textbook reader-writer lock scenario — not a simple serial queue, or your search latency degrades badly under concurrent load.

```swift
final class ReadWriteLock {
    private var lock = pthread_rwlock_t()
    init() { pthread_rwlock_init(&lock, nil) }
    deinit { pthread_rwlock_destroy(&lock) }

    func withRead<T>(_ body: () throws -> T) rethrows -> T {
        pthread_rwlock_rdlock(&lock)
        defer { pthread_rwlock_unlock(&lock) }
        return try body()
    }

    func withWrite<T>(_ body: () throws -> T) rethrows -> T {
        pthread_rwlock_wrlock(&lock)
        defer { pthread_rwlock_unlock(&lock) }
        return try body()
    }
}
```

Wrap the core engine in a public `actor` so callers get Swift Concurrency ergonomics (`await`), while the actor's internal implementation uses the `ReadWriteLock` around the actual `UnsafeMutablePointer` work for performance — an `actor` alone serializes *everything*, including reads, which is wasteful for a read-heavy search workload. Concretely:

```swift
public actor VectorDB {
    private let engine: Engine   // a plain (non-actor) class holding ReadWriteLock + indices

    public func search(query: [Float], k: Int) throws -> [SearchResult] {
        try engine.search(query: query, k: k)   // engine internally does withRead { ... }
    }
    public func insert(id: String, vector: [Float]) throws {
        try engine.insert(id: id, vector: vector) // engine internally does withWrite { ... }
    }
}
```

> Why not just make `Engine` itself the actor and skip the rwlock? Because actor isolation is single-threaded by design — you'd serialize concurrent searches behind each other unnecessarily. The outer `actor VectorDB` gives you a `Sendable`-safe, async-friendly public surface; the inner `pthread_rwlock_t` gives you real read-parallelism where it matters. This is a deliberate two-layer design, not redundancy.

### Pitfalls
- **Writer starvation**: `pthread_rwlock` on Darwin is not guaranteed fair — a steady stream of readers can starve a pending writer. If you observe this in testing (inserts stalling under heavy concurrent search), consider `pthread_rwlock_attr_setkind_np` variants or fall back to a simpler exclusive lock and accept the read-parallelism loss.
- **Long-held read locks during a slow disk load** (Phase 6): don't hold the rwlock while performing I/O; mmap access itself is fast (page faults are handled by the kernel), but any explicit file read/parsing should happen outside the lock, with the lock only guarding the in-memory index handoff.
- **Growing `VectorStorage`/`GraphStorage` under a read lock**: `grow()` mutates the buffer pointer itself — this absolutely must happen only under the write lock, never concurrently with any reader holding a pointer into the old buffer. This is the concurrency-flavored version of the Phase 3 dangling-pointer pitfall.

### Definition of Done
- [ ] Stress test: 8 concurrent reader tasks continuously searching while 1 writer task continuously inserts, run for 60 seconds under Thread Sanitizer, zero races reported.
- [ ] Confirm search latency under concurrent load is not dramatically worse than single-threaded (some degradation is expected and fine; a full lock-contention collapse is not).

---

## 10. Phase 6 — Persistence: File Format & Memory-Mapped I/O

### 10.1 Why mmap, specifically

Loading a 500MB index file by reading it fully into a Swift `Data`/`[Float]` on app launch is slow and spikes memory. `mmap` instead maps the file's pages lazily — the OS pages data in from the SSD only as your code actually touches those addresses, and the pages are shared/reclaimable under memory pressure. For a 100k-vector (384-dim, float32) collection, that's roughly `100,000 × 384 × 4 bytes ≈ 146MB` of vector data alone, plus graph overhead — exactly the regime where mmap matters on a memory-constrained phone.

### 10.2 File Format

Design a fixed layout so that any byte range can be addressed by arithmetic alone — never by scanning:

```
┌─────────────────────────────────────────────────────────┐
│ HEADER (fixed size, e.g. 256 bytes, padded)               │
│  - magic: "SVDB" (4 bytes)                                 │
│  - formatVersion: UInt32                                   │
│  - dimension: UInt32                                        │
│  - vectorCount: UInt64                                       │
│  - capacity: UInt64          (allocated slots, >= vectorCount)│
│  - metric: UInt8 (0=cosine,1=euclidean,2=dot)                │
│  - hnswM: UInt32, hnswMmax0: UInt32                           │
│  - entryPointID: Int32, entryPointLevel: UInt32               │
│  - vectorSectionOffset: UInt64                                │
│  - graphSectionOffset: UInt64                                  │
│  - idMapSectionOffset: UInt64                                    │
│  - checksum: UInt64 (of everything after the header — pick a fast   │
│    non-cryptographic hash such as CRC32C or a 64-bit FNV/xxHash     │
│    variant and document the choice; it only needs to catch          │
│    truncation/corruption, not resist tampering)                     │
├─────────────────────────────────────────────────────────┤
│ VECTOR SECTION                                             │
│  capacity * dimension * 4 bytes, row-major Float32          │
├─────────────────────────────────────────────────────────┤
│ GRAPH SECTION (layer 0)                                     │
│  capacity * Mmax0 * 4 bytes, Int32 neighbor ids, padded -1   │
├─────────────────────────────────────────────────────────┤
│ GRAPH SECTION (upper layers, variable — see note below)      │
├─────────────────────────────────────────────────────────┤
│ ID MAP SECTION                                              │
│  String id <-> Int32 internal id + per-vector metadata blob   │
│  (variable length — NOT mmap-addressed directly; read fully  │
│   into memory on load, it's small relative to vector data)    │
└─────────────────────────────────────────────────────────┘
```

> **Upper layers are inherently sparse** (most nodes never get promoted above layer 0), so pre-allocating fixed-width slots for every node at every possible layer wastes enormous space. Practical approach: store upper layers as a **separate small table** — for each node that has `level > 0`, store `(nodeID, level, [neighbor ids at each layer up to level])` in a compact variable-length format. Since this table is tiny relative to layer 0 and the vector data (by construction, only a `1/M`-ish fraction of nodes reach level 1, a `1/M²`-ish fraction reach level 2, etc.), it's fine to load this part fully into memory rather than mmap-address it directly — don't over-engineer the sparse case.

### 10.3 Swift mmap Wrapper

```swift
import Darwin

final class MappedFile {
    let fileDescriptor: Int32
    private(set) var size: Int
    private(set) var pointer: UnsafeMutableRawPointer

    init(path: String, initialSize: Int) throws {
        fileDescriptor = open(path, O_RDWR | O_CREAT, 0o644)
        guard fileDescriptor >= 0 else { throw VectorDBError.ioError(errno: errno) }

        var st = stat()
        fstat(fileDescriptor, &st)
        let currentSize = Int(st.st_size)
        self.size = max(currentSize, initialSize)
        if currentSize < self.size {
            guard ftruncate(fileDescriptor, off_t(self.size)) == 0 else {
                close(fileDescriptor)
                throw VectorDBError.ioError(errno: errno)
            }
        }

        guard let mapped = mmap(nil, self.size, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0),
              mapped != MAP_FAILED else {
            close(fileDescriptor)
            throw VectorDBError.ioError(errno: errno)
        }
        self.pointer = mapped
    }

    /// Grows the mapping when the index outgrows its current capacity.
    /// mmap on Darwin does NOT support in-place remap (no mremap like Linux) —
    /// must unmap, resize the file, and remap.
    func resize(to newSize: Int) throws {
        guard munmap(pointer, size) == 0 else { throw VectorDBError.ioError(errno: errno) }
        guard ftruncate(fileDescriptor, off_t(newSize)) == 0 else { throw VectorDBError.ioError(errno: errno) }
        guard let mapped = mmap(nil, newSize, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0),
              mapped != MAP_FAILED else { throw VectorDBError.ioError(errno: errno) }
        pointer = mapped
        size = newSize
    }

    /// Flush dirty pages to disk. Call after batched writes, not per-insert (too slow).
    func sync() {
        msync(pointer, size, MS_SYNC)
    }

    deinit {
        munmap(pointer, size)
        close(fileDescriptor)
    }
}
```

> **Critical Darwin-specific gotcha**: unlike Linux's `mremap`, macOS/iOS has no atomic "grow this mapping in place." Growing means unmap → ftruncate → remap, and the OS may hand you back a **different base address**. Every raw pointer you've handed out into the old mapping (e.g., anything `VectorStorage`/`GraphStorage` cached) is now invalid. **This is the same "never cache raw pointers across a mutation" rule from Phase 3, now with higher stakes.** The cleanest fix: have `VectorStorage`/`GraphStorage`, when backed by a `MappedFile`, always recompute their base pointer from `mappedFile.pointer` + a fixed offset rather than storing their own independent pointer — i.e. treat `MappedFile.pointer` as the single source of truth, queried fresh on each access, not cached.

### 10.4 Durability: Write-Ahead Log + Snapshot/Compaction

Writing every single insert straight into the mmap'd file and calling `msync` each time is durable but far too slow (a syscall + disk flush per insert). Instead:

1. **WAL**: append-only file, each record = `(opcode: insert/delete, internalID, vector bytes, timestamp)`. `fsync` the WAL periodically (e.g. every N inserts or every M milliseconds, developer-configurable) — this bounds your "data loss window" on a crash without paying the full flush cost per write.
2. **In-memory index** is always the source of truth for `search()` — the WAL exists purely for crash recovery, not for reads.
3. **Snapshot/compaction**: periodically (developer calls `save()`, or automatically after the WAL exceeds a size threshold), serialize the full current in-memory state into a **new** snapshot file, `fsync` it, then atomically `rename()` it over the old snapshot, then truncate the WAL to empty. `rename()` on the same filesystem/volume is atomic on Darwin — this is what protects you from a corrupted half-written snapshot if the app is killed mid-save.
4. **On launch**: `mmap` the last good snapshot, then replay any WAL records written after that snapshot's timestamp.

```swift
enum WALOpcode: UInt8 { case insert = 0, delete = 1 }

struct PersistenceManager {
    func save(index: HNSWIndex, to url: URL) throws {
        let tmpURL = url.appendingPathExtension("tmp")
        try writeSnapshot(index, to: tmpURL)   // full serialize, fsync
        _ = try FileManager.default.replaceItemAt(url, withItemAt: tmpURL) // atomic on same volume
        try walFile.truncate()
    }

    func load(from url: URL) throws -> HNSWIndex {
        var index = try readSnapshot(from: url)
        for record in try walFile.readAllRecords() {
            try apply(record, to: &index)   // replay
        }
        return index
    }
}
```

### Pitfalls
- **Checksum validation on load** — always verify the header checksum before trusting a snapshot; a partially-written file from a prior crash (before your atomic-rename protection existed, or from a bug) should be detected and rejected/recovered-from rather than silently loaded as garbage and crashing deep inside HNSW traversal later.
- **Format version mismatches** — bump `formatVersion` on any layout change, refuse to load files from newer/incompatible versions with a clear `VectorDBError.unsupportedFileVersion`, and consider a migration path for older versions rather than silently misreading bytes.
- **`O_CREAT` permission/quota errors** — disk-full and permission-denied are real scenarios on iOS (low storage devices are common); every file operation should propagate a typed error the app can show to the user, never a silent no-op.
- **Backgrounding during a write**: iOS can suspend your app mid-operation. Wrap `save()` in a `UIApplication.shared.beginBackgroundTask` (or the appropriate Swift Concurrency equivalent) if it's triggered from a context where the app might be about to background, so the OS gives you a few extra seconds to finish the atomic rename rather than killing you mid-write.
- **App Group / extension access**: if multiple app extensions (e.g., a share extension and the main app) need to read the same store, you need actual file locking (`flock`) beyond just this process's in-memory rwlock — the current design assumes single-process access. Flag this as a v2 concern if you need cross-process safety.

### Definition of Done
- [ ] Kill the app process (simulate crash) mid-insert-burst, relaunch, verify no data loss beyond the configured WAL flush interval, and verify no corruption.
- [ ] Kill the process mid-`save()` (e.g., by injecting a delay + `raise(SIGKILL)` in a test harness before the rename completes), verify the old snapshot is still intact and loadable.
- [ ] Load a deliberately truncated/corrupted file, verify a typed error is thrown, not a crash.
- [ ] Memory footprint measured via Instruments before/after switching from full-file-read to mmap, on a 100k-vector store — confirm meaningfully lower resident memory.

---

## 11. Phase 7 — Deletion & Updates

HNSW does not support cheap, correctness-preserving deletion natively — removing a node's edges can disconnect the layers above it from the rest of the graph.

### Recommended approach for v1: Tombstone + Periodic Rebuild

```swift
mutating func remove(internalID: Int32) throws {
    guard nodes[internalID] != nil else { throw VectorDBError.notFound }
    tombstoned.insert(internalID)
    // Leave the node's edges in place — other nodes can still traverse *through* it
    // during search, we simply filter it out of the final result set.
}

func search(query: UnsafePointer<Float>, k: Int, ef: Int?) -> [(id: Int32, score: Float)] {
    // Fetch more than k, then filter, since some of the top candidates may be tombstoned.
    let raw = rawSearch(query: query, k: k + tombstoned.count, ef: ef)
    return raw.filter { !tombstoned.contains($0.id) }.prefix(k).map { $0 }
}
```

- Track `tombstoned.count / nodes.count` and trigger a **full rebuild** (re-insert all live vectors into a fresh `HNSWIndex` in original insertion order, or better, sorted by original level to preserve graph quality) once the ratio crosses a threshold (e.g. 10–20%). Do this rebuild off the main actor, then atomically swap the new index in under the write lock.
- **Update** (re-inserting a vector for an existing ID with new content) = tombstone the old internal ID + insert a fresh one, don't try to mutate an existing node's vector/edges in place — HNSW's neighbor structure was built around the old vector's position in space and become stale otherwise.

### Pitfalls
- Tombstoned nodes still consume memory and still get traversed *through* — if a large fraction of your graph is tombstoned, both memory and search latency degrade even though `count` (live vectors) looks small. Surface this in the public API (`db.stats()` exposing `liveCount`, `tombstonedCount`) so the app developer can decide when to trigger a manual rebuild if the automatic threshold isn't aggressive enough for their use case.
- If the current `entryPoint` gets tombstoned, do **not** remove it from the graph structure (it's still needed for traversal), but do pick a different, non-tombstoned `entryPoint` candidate for the *next* insert's descent-from-top phase, chosen from the highest surviving non-tombstoned level.
- A full rebuild is O(N log N)-ish and will briefly spike CPU/battery — schedule it (e.g., during `save()`, or when the app is backgrounded/charging) rather than synchronously inside a user-facing `delete()` call.

### Definition of Done
- [ ] Delete 30% of a 20k-vector index, verify `search` never returns a tombstoned ID.
- [ ] Trigger a rebuild, verify recall@10 against the `FlatIndex` oracle recovers to baseline levels (tombstoning alone, without rebuild, will show fine recall since edges are intact — the real test is that rebuild doesn't *regress* it).
- [ ] Verify `entryPoint` reassignment logic with a test that specifically tombstones the current entry point, then performs further inserts and searches successfully.

---

## 12. Phase 8 — Public API Design

```swift
public actor VectorDB {
    public init(dimension: Int,
                metric: DistanceMetric = .cosine,
                parameters: HNSWParameters = .default,
                path: URL? = nil) throws

    public func insert(id: String, vector: [Float], metadata: [String: String]? = nil) throws
    public func batchInsert(_ items: [(id: String, vector: [Float], metadata: [String: String]?)]) throws
    public func search(query: [Float], k: Int, ef: Int? = nil) throws -> [SearchResult]
    public func delete(id: String) throws
    public func update(id: String, vector: [Float], metadata: [String: String]? = nil) throws

    public func save() async throws
    public func close() async

    public struct Stats: Sendable {
        public let liveCount: Int
        public let tombstonedCount: Int
        public let dimension: Int
        public let onDiskSizeBytes: Int?
    }
    public func stats() -> Stats
}

public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let metadata: [String: String]?
}

public enum VectorDBError: Error, Sendable {
    case dimensionMismatch(expected: Int, got: Int)
    case duplicateID(String)
    case notFound(String)
    case invalidVector(reason: String)      // NaN, Inf, zero-vector-for-cosine
    case ioError(errno: Int32)
    case corruptFile(reason: String)
    case unsupportedFileVersion(found: UInt32, supported: UInt32)
    case invalidParameters(reason: String)
}
```

### Decisions to make explicit in your API docs (not just your head)
- **Duplicate ID policy**: does `insert` with an existing ID throw `.duplicateID`, or silently upsert? Recommendation: throw, and provide an explicit `update()` for the upsert case — surprising silent overwrites are a common source of app-level bugs.
- **In-memory-only mode**: `path: nil` should give you a valid, fully-functional in-memory-only database (useful for ephemeral/session-scoped search, or testing) — don't force disk I/O as a hard requirement of the type.
- **Synchronous vs. async**: `insert`/`search` are CPU-bound, not I/O-bound (I/O happens async, in `save()`), so they can reasonably be synchronous `throws` functions called from within the actor's isolation, rather than `async throws`. Being deliberate about this avoids forcing every call site into unnecessary `Task { }` wrapping for pure in-memory operations.
- **`close()` semantics**: define this explicitly — at minimum it should flush any pending WAL records (best-effort final `save()`), `munmap`/close the underlying file descriptors (§10.3), and leave the actor in a state where subsequent calls either throw a clear `.closed`-style error or are documented as undefined; make `close()` itself safe to call more than once (idempotent), since app lifecycle callbacks can invoke cleanup paths more than once.
- **`batchInsert` locking**: acquire the write lock (§9) once for the entire batch rather than once per item — this is the whole point of exposing a batch API instead of calling `insert` in a loop from the app side. Decide and document whether a failure partway through the batch (e.g. one bad vector) aborts the whole batch (all-or-nothing) or applies items up to the failure and reports which ones failed.

---

## 13. Phase 9 — Testing Strategy

- **`FlatIndexTests`**: exact correctness of brute-force search against hand-computed expected results for small, fixed vectors (dimension 2–4, human-checkable by hand).
- **`HNSWCorrectnessTests`**: for every test dataset size (1k / 10k / 50k), build both a `FlatIndex` and `HNSWIndex` from the *same* vectors, run identical queries against both, and assert `recall@k ≥ threshold` (compute recall as `|HNSW_topK ∩ FlatIndex_topK| / k`, averaged over many queries — not a single query, which is noisy).
- **Deterministic fixtures**: a seeded random vector generator (`TestFixtures.swift`) so every test run is byte-for-byte reproducible.
- **`PersistenceTests`**: round-trip save/load equivalence (`search` results identical before save and after load); crash-simulation tests as described in §10's Definition of Done.
- **`ConcurrencyTests`**: run under Thread Sanitizer (`-sanitize=thread` scheme setting) with the concurrent reader/writer stress test from §9.
- **`EdgeCaseTests`**: every item in §16 below should have a corresponding test, not just a mental note.
- **Property-based-style fuzzing**: generate random sequences of insert/delete/search operations against both a `FlatIndex`-backed reference and the real `HNSWIndex`, assert search results stay consistent (within recall tolerance) after arbitrary operation interleavings — this catches state-machine bugs that hand-written scenario tests miss.

---

## 14. Phase 10 — Benchmarking & Tuning

Do **not** use `XCTest`'s `measure {}` for serious performance work — it's fine for regression-catching in CI but its statistical reporting is thin. Build a standalone executable target (`Benchmarks/VectorDBBenchmarks/main.swift`) that:

1. Loads/generates a dataset at realistic scale (start with synthetic random unit vectors, then validate again with real CoreML sentence-embedding output before shipping — random vectors have different clustering behavior than real semantic embeddings, and HNSW recall can differ between the two).
2. Sweeps `efSearch` values (e.g. 10, 25, 50, 100, 200) and plots the **recall-vs-latency curve** — this is the single most useful chart for choosing your shipped default.
3. Measures: build time (full index construction), p50/p95/p99 query latency, memory footprint (RSS via `mach_task_basic_info`), and — critically for this project's stated goals — **energy impact**, measured via Instruments' Energy Log template on a real device, not the simulator.
4. Compares HNSW against the `FlatIndex` brute-force baseline at the same dataset size, to have a concrete "why does this complexity exist" number for your own documentation/README.

Track these numbers in a checked-in `BENCHMARKS.md` alongside the code so regressions are visible in code review, not just discovered in production.

---

## 15. Phase 11 — Packaging, CI, Distribution

- **CI**: GitHub Actions (or equivalent) running `swift test` on both a Mac runner (native) — there's no meaningful CI story for iOS-only Accelerate code beyond building for the simulator/device destination, so also add a build-only job with `xcodebuild -scheme VectorDB -destination 'platform=iOS Simulator,name=iPhone 15'` to catch iOS-specific compile issues (e.g. accidental use of a macOS-only API).
- **Thread Sanitizer job**: a dedicated CI job running the concurrency test suite with `-sanitize=thread` — do not let this regress silently, it's easy to reintroduce a race when refactoring `GraphStorage`.
- **Semantic versioning**: bump major version on any on-disk format change (paired with the `formatVersion` field from §10.2) — a developer's shipped app updating your SDK should never silently corrupt their users' on-disk stores.
- **README**: lead with the 5-line usage example from §1, then link into this document (or a trimmed public version of it) for internals.

---

## 16. Edge Case & Failure Mode Checklist

Work through this list explicitly — each row should map to a real test, not just a mental note.

| Category | Case | Expected Behavior |
|---|---|---|
| Input validation | Vector dimension ≠ configured dimension | Throw `.dimensionMismatch` before touching unsafe storage |
| Input validation | Vector contains NaN/Inf | Throw `.invalidVector` |
| Input validation | Zero vector with `.cosine` metric | Throw `.invalidVector` (undefined similarity) or document a defined fallback (e.g. treat as zero similarity to everything) — pick one and be consistent |
| Input validation | Empty string ID | Decide policy (allow or reject) and document it |
| Insert | Duplicate ID | Throw `.duplicateID` (see §12 decision) |
| Insert | Insert into index at exactly `capacity` | Triggers `grow()`, verify no dangling pointer (§7) |
| Search | `k = 0` | Return `[]`, don't throw |
| Search | `k` > live vector count | Clamp and return all live vectors, don't throw or crash |
| Search | Search on empty index | Return `[]` |
| Search | Search on index with 100% tombstoned entries | Return `[]`, not a crash from an empty post-filter array |
| Delete | Delete non-existent ID | Throw `.notFound` |
| Delete | Delete the current `entryPoint` | Reassign entry point correctly (§11) |
| Delete | Delete all vectors, then insert again | Index must recover to a valid, searchable state — test this explicitly, it's a common "empty state" bug source |
| Persistence | Load a file with wrong magic bytes | Throw `.corruptFile`, not a crash |
| Persistence | Load a file from a newer format version | Throw `.unsupportedFileVersion` |
| Persistence | Disk full during `save()` | Propagate `.ioError`, leave the *previous* snapshot intact and loadable |
| Persistence | App killed mid-WAL-write | Recovery replay must skip/ignore the incomplete trailing record (length-prefix or checksum each WAL record individually) |
| Concurrency | Concurrent `insert` + `search` | No data race (verify under TSan), search sees either the fully-old or fully-new state, never a torn/partial write |
| Concurrency | Concurrent `save()` + `insert` | `save()` must snapshot a consistent point-in-time view; decide (and document) whether concurrent inserts during a save are blocked, queued, or included/excluded |
| Memory | Very low memory device / memory warning | mmap'd pages can be reclaimed by the OS under pressure — verify performance degrades gracefully (re-fault from disk) rather than crashing |
| Scale | Single-vector index | All graph logic must degrade correctly to trivial cases (level 0 only, no upper layers) |
| Scale | Extremely high-dimensional vectors (e.g. 4096-dim) | Confirm no `Int32` overflow in offset math (`slot * dimension` — use `Int` consistently, verify at your target max collection size that `count * dimension` doesn't approach `Int32` limits if you've used `Int32` anywhere in offset calculations) |

---

## 17. Glossary

- **Embedding**: a fixed-length float vector representing the semantic content of some input (text/image), produced by a model such as a CoreML sentence encoder.
- **Cosine similarity**: a measure of the angle between two vectors, ignoring magnitude; the standard metric for comparing semantic embeddings.
- **HNSW**: Hierarchical Navigable Small World graph — an approximate nearest-neighbor search structure using multiple layers of proximity graphs.
- **Recall@k**: the fraction of the true top-k nearest neighbors (from exhaustive search) that an approximate method actually returns.
- **`ef` (efConstruction / efSearch)**: the size of the dynamic candidate list HNSW maintains during graph construction / query time; the primary recall/speed tuning knob.
- **mmap (memory-mapped file)**: an OS mechanism that maps a file's contents directly into a process's virtual address space, letting the kernel page data in/out on demand instead of requiring an explicit read into a memory buffer.
- **WAL (Write-Ahead Log)**: an append-only durability log; changes are recorded here before/instead of being applied to the main data file synchronously, then periodically compacted.
- **vDSP**: part of Apple's Accelerate framework; a library of SIMD-accelerated vector/signal-processing primitives (dot products, sums, distances, etc.).
- **BLAS (Basic Linear Algebra Subprograms)**: a standard API for matrix/vector operations; Accelerate provides an Apple-optimized implementation (`cblas_*` functions) used here for batched similarity computation.
- **Tombstone**: a soft-delete marker; the record is logically removed but its underlying structure is left in place until a later compaction/rebuild physically removes it.

---

## 18. References

- Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI. — the primary HNSW paper; read this alongside Phase 4.
- Apple Developer Documentation: **Accelerate framework**, specifically the `vDSP` and `BLAS` sections — for exact function signatures as APIs evolve across OS versions, check the current docs rather than relying solely on this guide's snippets.
- Apple Developer Documentation: **`mmap`, `Foundation.Data(contentsOf:options:)`** — for the higher-level (non-raw-syscall) mmap option, useful for the read-only load path if you want to avoid raw Darwin calls where write access isn't needed.
- `apple/swift-collections` — if you choose the non-zero-dependency path for `Heap`/`Deque` (§2).

---

## 19. Milestone Checklist

- [ ] **M0** — Package scaffolded, builds on iOS + macOS destinations, empty test target runs.
- [ ] **M1** — `FlatIndex` complete: insert/search/remove correct on synthetic data, all three metrics tested.
- [ ] **M2** — `VectorMath` ported to Accelerate/vDSP, benchmarked against the naive loop, cosine pre-normalization implemented.
- [ ] **M3** — `VectorStorage` unsafe contiguous buffer complete, growth-safety stress-tested, zero leaks in Instruments.
- [ ] **M4** — `HNSWIndex` insert + search complete, recall@10 ≥ 0.95 against `FlatIndex` oracle at 50k vectors.
- [ ] **M5** — Concurrency layer (`ReadWriteLock` + actor wrapper) complete, zero TSan warnings under stress test.
- [ ] **M6** — File format finalized, `MappedFile` complete, WAL + snapshot/compaction implemented, crash-recovery tests pass.
- [ ] **M7** — Tombstone deletion + rebuild implemented and tested.
- [ ] **M8** — Public `VectorDB` actor API finalized and documented; error taxonomy complete.
- [ ] **M9** — Full edge-case checklist (§16) covered by tests.
- [ ] **M10** — Benchmark suite complete, `BENCHMARKS.md` published with real device numbers, `efSearch` default chosen from real recall/latency data (not guessed).
- [ ] **M11** — CI (build + test + TSan job) green, versioning policy documented, README written, package tagged `1.0.0`.

---

*End of guide. Update this file as design decisions change — it should stay the living source of truth for the project, not a snapshot of day-one intentions.*
