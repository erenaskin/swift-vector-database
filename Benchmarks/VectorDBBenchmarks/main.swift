/// main.swift — Benchmarks & Leak Tests.
///
/// Run in release mode for meaningful numbers:
///   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
///   swift run -c release VectorDBBenchmarks

import Foundation
import Accelerate
import NaturalLanguage
import VectorDB

// MARK: - Helper: high-resolution wall clock

func wallClock(_ label: String, _ block: () -> Void) -> Double {
    let start = ProcessInfo.processInfo.systemUptime
    block()
    let elapsed = ProcessInfo.processInfo.systemUptime - start
    print("\(label): \(String(format: "%.4f", elapsed * 1000.0))ms")
    return elapsed
}

// ──────────────────────────────────────────────────────────────────────
// Phase 2 Benchmark: vDSP_dotpr vs. Naive scalar loop (100k × 384-dim)
// ──────────────────────────────────────────────────────────────────────
func runPhase2Benchmarks() {
    let dim        = 384
    let iterations = 100_000

    let vecA = UnsafeMutablePointer<Float>.allocate(capacity: dim)
    let vecB = UnsafeMutablePointer<Float>.allocate(capacity: dim)
    defer { vecA.deallocate(); vecB.deallocate() }

    for i in 0..<dim {
        vecA[i] = Float(i + 1) / Float(dim)
        vecB[i] = Float(dim - i) / Float(dim)
    }

    print("=== Phase 2 Benchmark ===")
    print("Vector dim: \(dim)  |  Iterations: \(iterations)\n")

    var warmup: Float = 0
    for _ in 0..<1000 { vDSP_dotpr(vecA, 1, vecB, 1, &warmup, vDSP_Length(dim)) }

    var vdspResult: Float = 0
    let vdspTime = wallClock("vDSP_dotpr (100k iters)") {
        for _ in 0..<iterations {
            vDSP_dotpr(vecA, 1, vecB, 1, &vdspResult, vDSP_Length(dim))
        }
    }

    var naiveResult: Float = 0
    let naiveTime = wallClock("Naive scalar loop (100k iters)") {
        for _ in 0..<iterations {
            var sum: Float = 0
            for i in 0..<dim { sum += vecA[i] * vecB[i] }
            naiveResult = sum
        }
    }

    let speedup = naiveTime / vdspTime
    print("\nvDSP result : \(vdspResult)")
    print("Naive result: \(naiveResult)")
    print("Results match: \(abs(vdspResult - naiveResult) < 1e-4)")
    print("\n>>> Speedup: \(String(format: "%.2f", speedup))x  (naive / vDSP)")

    print("\n=== Batch Benchmark: sgemv vs. looped vDSP_dotpr ===")
    let batchCount = 1000
    let batchIters = 1000

    let matrix = UnsafeMutablePointer<Float>.allocate(capacity: batchCount * dim)
    defer { matrix.deallocate() }
    for i in 0..<batchCount * dim {
        matrix[i] = Float(i % dim + 1) / Float(dim)
    }
    var batchResults = [Float](repeating: 0, count: batchCount)

    let sgemvTime = wallClock("cblas_sgemv (\(batchCount) vecs, \(batchIters) iters)") {
        for _ in 0..<batchIters {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        Int32(batchCount), Int32(dim),
                        1.0, matrix, Int32(dim),
                        vecA, 1,
                        0.0, &batchResults, 1)
        }
    }

    var loopResult: Float = 0
    let loopTime = wallClock("Looped vDSP_dotpr (\(batchCount) vecs, \(batchIters) iters)") {
        for _ in 0..<batchIters {
            for j in 0..<batchCount {
                vDSP_dotpr(vecA, 1, matrix + j * dim, 1, &loopResult, vDSP_Length(dim))
            }
        }
    }
    print("\n>>> Batch speedup: \(String(format: "%.2f", loopTime / sgemvTime))x  (looped vDSP / sgemv)")
}

// ──────────────────────────────────────────────────────────────────────
// Phase 3 Benchmark: 1M inserts for Instruments Leak test (§7)
// ──────────────────────────────────────────────────────────────────────
func runPhase3MemoryLeakTest() {
    print("\n=== Phase 3: 1M Vectors Insert Memory Test ===")
    let dim = 384
    let count = 1_000_000
    
    // We allocate a dummy vector to reuse for insert, avoiding swift array
    // alloc overhead so we strictly measure VectorStorage memory.
    let dummyVector = UnsafeMutablePointer<Float>.allocate(capacity: dim)
    defer { dummyVector.deallocate() }
    for i in 0..<dim { dummyVector[i] = Float(i) }
    
    // Memory size of 1M vectors of 384 dim = 1M * 384 * 4 bytes = 1.5GB.
    
    let storage = VectorStorage(dimension: dim, initialCapacity: 1024)
    
    _ = wallClock("Insert 1,000,000 vectors (dim=384)") {
        for _ in 0..<count {
            storage.append(dummyVector)
        }
    }
    print("Storage capacity reached: \(storage.capacity) vectors")
    print("Storage count reached: \(storage.count) vectors")
    print("Check memory usage in Activity Monitor / Instruments now.")
    print("Sleeping for 3 seconds to allow inspection...")
    Thread.sleep(forTimeInterval: 3.0)
    print("Done.")
}

// ──────────────────────────────────────────────────────────────────────
// Phase 4 Benchmark §8 DoD: 50k insert timing + recall@10 vs. FlatIndex
// ──────────────────────────────────────────────────────────────────────
func runPhase4Benchmarks() {
    print("\n=== Phase 4 Benchmark (§8 Definition of Done) ===")

    let dim     = 64    // Reduced dim for benchmark speed; recall holds at 64-dim
    let count   = 50_000
    let k       = 10
    let queries = 500   // Number of query vectors for recall measurement

    // ---- Seeded vector generation (no randomness between runs) ----
    func makeVectors(_ n: Int, seed: UInt64) -> [[Float]] {
        var rng = SeedableRNG(seed: seed)
        return (0..<n).map { _ in
            let v = (0..<dim).map { _ -> Float in
                return rng.nextFloat()
            }
            let norm = sqrt(v.reduce(Float(0)) { $0 + $1 * $1 })
            return norm > 1e-6 ? v.map { $0 / norm } : v
        }
    }

    print("Generating \(count) unit vectors (dim=\(dim))...")
    let corpus  = makeVectors(count, seed: 0xDEAD_BEEF_0001)
    let qvecs   = makeVectors(queries, seed: 0xCAFE_BABE_0002)

    // ── DoD item 2: Insert timing ──
    print("\nBuilding HNSWIndex (M=16, efConstruction=200)...")
    let params = HNSWParameters(M: 16, efConstruction: 200, efSearch: 300, seed: 0x5EED_1234)
    var hnsw = HNSWIndex(dimension: dim, metric: .cosine, params: params)

    let insertTime = wallClock("Insert \(count) vectors (dim=\(dim))") {
        for (i, v) in corpus.enumerated() {
            v.withUnsafeBufferPointer { buf in
                try? hnsw.insert(internalID: Int32(i), vector: buf.baseAddress!)
            }
        }
    }
    print("→ Insert time: \(String(format: "%.1f", insertTime * 1000))ms  (\(String(format: "%.1f", Double(count) / insertTime / 1000))k vectors/s)")

    // ── DoD item 1: Recall@10 vs. FlatIndex oracle ──
    print("\nBuilding FlatIndex oracle...")
    var flat = FlatIndex(dimension: dim, metric: .cosine)
    for (i, v) in corpus.enumerated() {
        v.withUnsafeBufferPointer { buf in
            try? flat.insert(internalID: Int32(i), vector: buf.baseAddress!)
        }
    }

    print("Measuring Recall@\(k) over \(queries) queries (efSearch=\(params.efSearch))...")
    var hits = 0
    var totalQueryTimeHNSW = 0.0
    var totalQueryTimeFlat = 0.0

    var firstPrint = true
    for qv in qvecs {
        let hnswResult: [(Int32, Float)] = qv.withUnsafeBufferPointer { buf in
            let t0 = ProcessInfo.processInfo.systemUptime
            let r = hnsw.search(query: buf.baseAddress!, k: k)
            totalQueryTimeHNSW += ProcessInfo.processInfo.systemUptime - t0
            return r
        }
        let flatResult: [(Int32, Float)] = qv.withUnsafeBufferPointer { buf in
            let t0 = ProcessInfo.processInfo.systemUptime
            let r = flat.search(query: buf.baseAddress!, k: k)
            totalQueryTimeFlat += ProcessInfo.processInfo.systemUptime - t0
            return r
        }
        if firstPrint {
            print("First query HNSW: \(hnswResult)")
            print("First query Flat: \(flatResult)")
            firstPrint = false
        }
        let hnswSet = Set(hnswResult.map(\.0))
        hits += flatResult.map(\.0).filter { hnswSet.contains($0) }.count
    }

    let totalRelevant = queries * k
    let recall = Double(hits) / Double(totalRelevant)

    print("→ Recall@\(k): \(String(format: "%.4f", recall)) (\(hits)/\(totalRelevant) hits)")
    print("→ HNSW avg query latency: \(String(format: "%.3f", totalQueryTimeHNSW / Double(queries) * 1000.0))ms")
    print("→ FlatIndex avg query latency: \(String(format: "%.3f", totalQueryTimeFlat / Double(queries) * 1000.0))ms")
    print("→ HNSW/Flat query speedup: \(String(format: "%.1f", totalQueryTimeFlat / totalQueryTimeHNSW))x")

    let dodStatus = recall >= 0.95 ? "✅ PASS" : "❌ FAIL (below 0.95 threshold)"
    print("\n§8 DoD Recall@\(k) check: \(dodStatus)  (actual: \(String(format: "%.4f", recall)))")

    // ── DoD item 3: Deterministic graph construction end-to-end ──
    print("\nVerifying end-to-end deterministic graph construction...")
    var hnsw2 = HNSWIndex(dimension: dim, metric: .cosine, params: params)
    for (i, v) in corpus.enumerated() {
        v.withUnsafeBufferPointer { buf in
            try? hnsw2.insert(internalID: Int32(i), vector: buf.baseAddress!)
        }
    }

    // Compare neighbor lists for first 1000 nodes (representative sample)
    var determinismOK = true
    outer: for id in 0..<Int32(min(1000, count)) {
        for layer in 0..<5 {
            let n1 = hnsw.neighbors(of: id, at: layer)
            let n2 = hnsw2.neighbors(of: id, at: layer)
            if n1 != n2 {
                print("  ❌ Node \(id) layer \(layer) differs: \(n1) vs \(n2)")
                determinismOK = false
                break outer
            }
        }
    }
    print("→ Determinism (1000-node sample): \(determinismOK ? "✅ byte-identical" : "❌ DIFFERS")")
}

// MARK: - Main Execution

runPhase2Benchmarks()
runPhase3MemoryLeakTest()
runPhase4Benchmarks()
runPhase10Benchmarks()
print("\n=== All Benchmarks Complete ===")

// ──────────────────────────────────────────────────────────────────────
// Phase 10 Benchmark: Performance, Tuning & Real NLP Embeddings
// ──────────────────────────────────────────────────────────────────────

func reportRSS(label: String) {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        print("\(label) RSS: \(info.resident_size / 1024 / 1024) MB")
    }
}

func percentiles(of latencies: [Double]) -> (p50: Double, p95: Double, p99: Double) {
    let sorted = latencies.sorted()
    guard !sorted.isEmpty else { return (0, 0, 0) }
    func p(_ pct: Double) -> Double {
        let idx = Int(Double(sorted.count - 1) * pct)
        return sorted[idx]
    }
    return (p(0.50), p(0.95), p(0.99))
}

func runPhase10Benchmarks() {
    print("\n=== Phase 10 Benchmark: Benchmarking & Tuning ===")
    
    reportRSS(label: "Initial")
    
    // 1. Dataset Generation
    let count = 10_000
    let syntheticDim = 384
    
    // Synthetic
    var rng = SeedableRNG(seed: 0x12345678)
    let syntheticCorpus = (0..<count).map { _ -> [Float] in
        let v = (0..<syntheticDim).map { _ in rng.nextFloat() }
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        return norm > 1e-6 ? v.map { $0 / norm } : v
    }
    let syntheticQueries = Array(syntheticCorpus.prefix(100))
    
    // Real (NaturalLanguage)
    var realCorpus: [[Float]] = []
    if let embedding = NLEmbedding.sentenceEmbedding(for: .english) {
        print("\nGenerating 10k real sentence embeddings using NaturalLanguage (this may take a moment)...")
        let subjects = ["The quick brown fox", "A lazy dog", "Some programmers", "Many scientists", "A wild tiger", "An elegant swan", "The database engine", "A swift compiler"]
        let verbs = ["jumps over", "analyzes", "sleeps near", "runs around", "calculates", "optimizes", "flies over", "ignores"]
        let objects = ["the fence", "the database", "a tree", "the quantum computer", "a river", "the memory leak", "a tall building", "the compiler error"]
        
        for i in 0..<count {
            let text = "\(subjects[i % subjects.count]) \(verbs[(i/subjects.count) % verbs.count]) \(objects[(i/(subjects.count*verbs.count)) % objects.count]) \(i)"
            if let vec = embedding.vector(for: text) {
                let floats = vec.map { Float($0) }
                let norm = sqrt(floats.reduce(0) { $0 + $1 * $1 })
                realCorpus.append(norm > 1e-6 ? floats.map { $0 / norm } : floats)
            }
        }
    } else {
        print("\nWarning: NLEmbedding.sentenceEmbedding is not available on this system.")
    }
    
    // 2. Benchmark Runner
    func runBenchmark(name: String, corpus: [[Float]], queries: [[Float]]) {
        print("\n--- Dataset: \(name) (\(corpus.count) vectors, dim=\(corpus[0].count)) ---")
        let dim = corpus[0].count
        let k = 10
        
        // Build FlatIndex
        var flat = FlatIndex(dimension: dim, metric: .cosine)
        _ = wallClock("Build FlatIndex (\(name))") {
            for (i, v) in corpus.enumerated() {
                v.withUnsafeBufferPointer { try! flat.insert(internalID: Int32(i), vector: $0.baseAddress!) }
            }
        }
        reportRSS(label: "After FlatIndex")
        
        var flatGroundTruth: [[Int32]] = []
        var flatLatencies: [Double] = []
        
        for q in queries {
            let t0 = ProcessInfo.processInfo.systemUptime
            let res = q.withUnsafeBufferPointer { flat.search(query: $0.baseAddress!, k: k) }
            flatLatencies.append((ProcessInfo.processInfo.systemUptime - t0) * 1000)
            flatGroundTruth.append(res.map { $0.0 })
        }
        let fPcts = percentiles(of: flatLatencies)
        print("FlatIndex Latency (ms): p50=\(String(format: "%.3f", fPcts.p50)), p95=\(String(format: "%.3f", fPcts.p95)), p99=\(String(format: "%.3f", fPcts.p99))")
        
        // Build HNSWIndex
        let params = HNSWParameters(M: 16, efConstruction: 200, efSearch: 64, seed: 0x5EED)
        var hnsw = HNSWIndex(dimension: dim, metric: .cosine, params: params)
        _ = wallClock("Build HNSWIndex (\(name))") {
            for (i, v) in corpus.enumerated() {
                v.withUnsafeBufferPointer { try! hnsw.insert(internalID: Int32(i), vector: $0.baseAddress!) }
            }
        }
        reportRSS(label: "After HNSWIndex")
        
        print("efSearch Sweep for HNSWIndex:")
        let sweeps = [10, 25, 50, 100, 200]
        for ef in sweeps {
            var hits = 0
            var hnswLatencies: [Double] = []
            
            for (qi, q) in queries.enumerated() {
                let t0 = ProcessInfo.processInfo.systemUptime
                let res = q.withUnsafeBufferPointer { hnsw.search(query: $0.baseAddress!, k: k, ef: ef) }
                hnswLatencies.append((ProcessInfo.processInfo.systemUptime - t0) * 1000)
                
                let hnswSet = Set(res.map { $0.0 })
                hits += flatGroundTruth[qi].filter { hnswSet.contains($0) }.count
            }
            
            let recall = Double(hits) / Double(queries.count * k)
            let hPcts = percentiles(of: hnswLatencies)
            print("  efSearch=\(String(format: "%3d", ef)) | Recall@\(k): \(String(format: "%.4f", recall)) | Latency (ms): p50=\(String(format: "%.3f", hPcts.p50)), p95=\(String(format: "%.3f", hPcts.p95)), p99=\(String(format: "%.3f", hPcts.p99))")
        }
    }
    
    runBenchmark(name: "Synthetic Random", corpus: syntheticCorpus, queries: syntheticQueries)
    if !realCorpus.isEmpty {
        runBenchmark(name: "NaturalLanguage Real", corpus: realCorpus, queries: Array(realCorpus.prefix(100)))
    }
}
