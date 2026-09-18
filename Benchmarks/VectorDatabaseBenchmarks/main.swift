/// main.swift — Benchmarks & Leak Tests.
///
/// Run in release mode for meaningful numbers:
///   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
///   swift run -c release VectorDatabaseBenchmarks

import Accelerate
import Foundation
import NaturalLanguage
import VectorDatabase

// MARK: - Helper: high-resolution wall clock

func wallClock(_ label: String, _ block: () -> Void) -> Double {
    let start = ProcessInfo.processInfo.systemUptime
    block()
    let elapsed = ProcessInfo.processInfo.systemUptime - start
    print("\(label): \(String(format: "%.4f", elapsed * 1000.0))ms")
    return elapsed
}

func reportRSS(label: String) {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        print("\(label) RSS: \(info.resident_size / 1024 / 1024) MB")
    }
}

// ──────────────────────────────────────────────────────────────────────
// Benchmark: vDSP_dotpr vs. Naive scalar loop (100k × 384-dim)
// ──────────────────────────────────────────────────────────────────────
func benchmarkVectorMathAccelerators() {
    let dim = 384
    let iterations = 100_000

    let vecA = UnsafeMutablePointer<Float>.allocate(capacity: dim)
    let vecB = UnsafeMutablePointer<Float>.allocate(capacity: dim)
    defer {
        vecA.deallocate()
        vecB.deallocate()
    }

    for i in 0..<dim {
        vecA[i] = Float(i + 1) / Float(dim)
        vecB[i] = Float(dim - i) / Float(dim)
    }

    print("=== Benchmark: vDSP_dotpr vs. Naive scalar loop (100k × 384-dim) ===")
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
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
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
    print(
        "\n>>> Batch speedup: \(String(format: "%.2f", loopTime / sgemvTime))x  (looped vDSP / sgemv)"
    )
}

// ──────────────────────────────────────────────────────────────────────
// Benchmark: 1M inserts for Instruments Leak test
// ──────────────────────────────────────────────────────────────────────
func benchmarkVectorStorageMemoryAndLeaks() {
    print("\n=== Benchmark: 1M inserts for Instruments Leak test ===")
    let dim = 384
    let count = 1_000_000

    let dummyVector = UnsafeMutablePointer<Float>.allocate(capacity: dim)
    defer { dummyVector.deallocate() }
    for i in 0..<dim { dummyVector[i] = Float(i) }

    let storage = VectorStorage(dimension: dim, initialCapacity: 1024)

    _ = wallClock("Insert 1,000,000 vectors (dim=384)") {
        for _ in 0..<count {
            storage.append(dummyVector)
        }
    }
    print("Storage capacity reached: \(storage.capacity) vectors")
    print("Storage count reached: \(storage.count) vectors")
}

// ──────────────────────────────────────────────────────────────────────
// Benchmark DoD: 50k insert timing + recall@10 vs. FlatIndex
// ──────────────────────────────────────────────────────────────────────
func benchmarkHNSWIndexRecallAndDeterminism() {
    print("\n=== Benchmark DoD: 50k insert timing + recall@10 vs. FlatIndex ===")

    let dim = 64
    let count = 50_000
    let k = 10
    let queries = 500

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
    let corpus = makeVectors(count, seed: 0xDEAD_BEEF_0001)
    let qvecs = makeVectors(queries, seed: 0xCAFE_BABE_0002)

    print("\nBuilding HNSWIndex (M=16, efConstruction=200)...")
    let params = HNSWParameters(M: 16, efConstruction: 200, efSearch: 300, seed: 0x5EED_1234)
    var hnsw = HNSWIndex(dimension: dim, metric: .cosine, params: params)

    let insertTime = wallClock("Insert \(count) vectors (dim=\(dim))") {
        for (i, v) in corpus.enumerated() {
            v.withUnsafeBufferPointer {
                try? hnsw.insert(internalID: Int32(i), vector: $0.baseAddress!)
            }
        }
    }
    print(
        "→ Insert time: \(String(format: "%.1f", insertTime * 1000))ms  (\(String(format: "%.1f", Double(count) / insertTime / 1000))k vectors/s)"
    )

    print("Building FlatIndex oracle...")
    var flat = FlatIndex(dimension: dim, metric: .cosine)
    for (i, v) in corpus.enumerated() {
        v.withUnsafeBufferPointer {
            try? flat.insert(internalID: Int32(i), vector: $0.baseAddress!)
        }
    }

    print("Measuring Recall@\(k) over \(queries) queries (efSearch=\(params.efSearch))...")
    var hits = 0
    var totalQueryTimeHNSW = 0.0
    var totalQueryTimeFlat = 0.0

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

        let hnswSet = Set(hnswResult.map(\.0))
        hits += flatResult.map(\.0).filter { hnswSet.contains($0) }.count
    }

    let totalRelevant = queries * k
    let recall = Double(hits) / Double(totalRelevant)
    let dodStatus = recall >= 0.95 ? "✅ PASS" : "❌ FAIL (below 0.95 threshold)"
    print("DoD Recall@\(k) check: \(dodStatus)  (actual: \(String(format: "%.4f", recall)))")

    print("\nVerifying end-to-end deterministic graph construction...")
    var hnsw2 = HNSWIndex(dimension: dim, metric: .cosine, params: params)
    for (i, v) in corpus.enumerated() {
        v.withUnsafeBufferPointer {
            try? hnsw2.insert(internalID: Int32(i), vector: $0.baseAddress!)
        }
    }

    var determinismOK = true
    outer: for id in 0..<Int32(min(1000, count)) {
        for layer in 0..<5 {
            let n1 = hnsw.neighbors(of: id, at: layer)
            let n2 = hnsw2.neighbors(of: id, at: layer)
            if n1 != n2 {
                determinismOK = false
                break outer
            }
        }
    }
    print("→ Determinism (1000-node sample): \(determinismOK ? "✅ byte-identical" : "❌ DIFFERS")")
}

// ──────────────────────────────────────────────────────────────────────
// Benchmark: HNSW Scale Advantage (HNSW vs Flat at 500k vectors)
// ──────────────────────────────────────────────────────────────────────
func benchmarkHNSWScaleAdvantage() {
    print("\n=== Benchmark: HNSW Scale Advantage (500k Vectors) ===")

    let dim = 128
    let count = 500_000
    let queries = 500
    let k = 10

    var rng = SeedableRNG(seed: 0x9999_8888)
    func randomVector() -> [Float] {
        let v = (0..<dim).map { _ in rng.nextFloat() }
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        return norm > 1e-6 ? v.map { $0 / norm } : v
    }

    print("Generating \(count) random vectors (dim=\(dim))...")
    let corpus = (0..<count).map { _ in randomVector() }
    let qvecs = (0..<queries).map { _ in randomVector() }

    // 1. Build Flat Index
    var flat = FlatIndex(dimension: dim, metric: .cosine)
    print("Building FlatIndex...")
    for (i, v) in corpus.enumerated() {
        v.withUnsafeBufferPointer {
            try! flat.insert(internalID: Int32(i), vector: $0.baseAddress!)
        }
    }

    // 2. Build HNSW Index (Zorlu ve Mükemmel Formül)
    // M=32 (Daha çok köprü), efConstruction=200 (Çok daha hassas inşa)
    let params = HNSWParameters(M: 32, efConstruction: 200, efSearch: 100, seed: 0x1111)
    var hnsw = HNSWIndex(dimension: dim, metric: .cosine, params: params)

    print("Building HNSWIndex...")
    let insertTime = wallClock("HNSW Insert 500k vectors") {
        for (i, v) in corpus.enumerated() {
            v.withUnsafeBufferPointer {
                try! hnsw.insert(internalID: Int32(i), vector: $0.baseAddress!)
            }
        }
    }
    print("HNSW Insert Rate: \(String(format: "%.1f", Double(count) / insertTime)) vectors/sec")

    // 3. Compare Latencies
    print("\nRunning \(queries) queries to compare speeds...")

    var flatTime: Double = 0
    let flatStart = ProcessInfo.processInfo.systemUptime
    for q in qvecs {
        _ = q.withUnsafeBufferPointer { flat.search(query: $0.baseAddress!, k: k) }
    }
    flatTime = ProcessInfo.processInfo.systemUptime - flatStart

    var hnswTime: Double = 0
    let hnswStart = ProcessInfo.processInfo.systemUptime
    for q in qvecs {
        _ = q.withUnsafeBufferPointer { hnsw.search(query: $0.baseAddress!, k: k) }
    }
    hnswTime = ProcessInfo.processInfo.systemUptime - hnswStart

    let avgFlatMs = (flatTime / Double(queries)) * 1000.0
    let avgHnswMs = (hnswTime / Double(queries)) * 1000.0

    print("→ FlatIndex avg query latency: \(String(format: "%.3f", avgFlatMs)) ms")
    print("→ HNSW avg query latency:      \(String(format: "%.3f", avgHnswMs)) ms")
    print(
        "→ HNSW Speedup at 500k scale: \(String(format: "%.1f", avgFlatMs / avgHnswMs))x FASTER than Flat"
    )
}

// ──────────────────────────────────────────────────────────────────────
// Benchmark: Hard NLP (Dense Semantic Space)
// ──────────────────────────────────────────────────────────────────────
func benchmarkHardNLPParameterSweeps() {
    print("\n=== Benchmark: Hard NLP (Dense Semantic Space - ULTIMATE CONFIG) ===")
    reportRSS(label: "Initial")

    var realCorpus: [[Float]] = []

    if let embedding = NLEmbedding.sentenceEmbedding(for: .english) {
        print("Generating 10k highly similar sentence embeddings (IT Incident Logs)...")

        let components = [
            "The user interface", "The backend server", "The database engine", "The network layer",
            "The authentication service", "The caching proxy", "The machine learning model",
            "The analytics pipeline", "The payment gateway", "The message queue",
        ]
        let actions = [
            "experiences high latency", "consumes excess memory", "drops active connections",
            "fails intermittently", "rejects valid payloads", "triggers deadlock states",
            "slows down drastically", "corrupts incoming data", "loses synchronization",
            "causes cpu throttling",
        ]
        let conditions = [
            "during peak traffic hours", "under heavy concurrent load",
            "when physical memory is low", "after the recent deployment",
            "on legacy mobile devices", "without proper cache warming", "during background tasks",
            "in offline failover mode", "across multiple active threads",
            "due to firewall restrictions",
        ]
        let results = [
            "requiring a manual restart.", "triggering an automated alert.",
            "which degrades user experience.", "but recovers automatically.",
            "necessitating data migration.", "leading to potential data loss.",
            "bypassing the load balancer.", "as a result of a timeout.",
            "blocking further requests.", "creating a bottleneck.",
        ]

        for comp in components {
            for act in actions {
                for cond in conditions {
                    for res in results {
                        let text = "\(comp) \(act) \(cond) \(res)"
                        if let vec = embedding.vector(for: text) {
                            let floats = vec.map { Float($0) }
                            let norm = sqrt(floats.reduce(0) { $0 + $1 * $1 })
                            realCorpus.append(norm > 1e-6 ? floats.map { $0 / norm } : floats)
                        }
                    }
                }
            }
        }
    } else {
        print("Warning: NLEmbedding not available.")
        return
    }

    let dim = realCorpus[0].count
    let k = 10
    let queries = Array(realCorpus.shuffled().prefix(200))  // 200 hard queries

    print("Dataset generated: \(realCorpus.count) vectors, dim=\(dim). Building FlatIndex...")
    var flat = FlatIndex(dimension: dim, metric: .cosine)
    for (i, v) in realCorpus.enumerated() {
        v.withUnsafeBufferPointer {
            try! flat.insert(internalID: Int32(i), vector: $0.baseAddress!)
        }
    }

    print("Calculating Ground Truth...")
    var flatGroundTruth: [[Int32]] = []
    for q in queries {
        let res = q.withUnsafeBufferPointer { flat.search(query: $0.baseAddress!, k: k) }
        flatGroundTruth.append(res.map { $0.0 })
    }

    print("Building HNSWIndex (M=32, efConstruction=300)...")
    let params = HNSWParameters(M: 32, efConstruction: 300, efSearch: 100, seed: 0x5EED)
    var hnsw = HNSWIndex(dimension: dim, metric: .cosine, params: params)
    for (i, v) in realCorpus.enumerated() {
        v.withUnsafeBufferPointer {
            try! hnsw.insert(internalID: Int32(i), vector: $0.baseAddress!)
        }
    }
    reportRSS(label: "After Index Builds")

    print("\nefSearch Sweep:")

    let sweeps = [10, 20, 40, 80, 150, 300]

    for ef in sweeps {
        var hits = 0
        var latencies: [Double] = []

        for (qi, q) in queries.enumerated() {
            let t0 = ProcessInfo.processInfo.systemUptime
            let res = q.withUnsafeBufferPointer {
                hnsw.search(query: $0.baseAddress!, k: k, ef: ef)
            }
            latencies.append((ProcessInfo.processInfo.systemUptime - t0) * 1000)

            let hnswSet = Set(res.map { $0.0 })
            hits += flatGroundTruth[qi].filter { hnswSet.contains($0) }.count
        }

        let recall = Double(hits) / Double(queries.count * k)
        let sortedLats = latencies.sorted()

        let p50 = sortedLats.isEmpty ? 0 : sortedLats[Int(Double(sortedLats.count) * 0.50)]
        let p95 = sortedLats.isEmpty ? 0 : sortedLats[Int(Double(sortedLats.count) * 0.95)]

        print(
            "  efSearch=\(String(format: "%3d", ef)) | Recall@\(k): \(String(format: "%.4f", recall)) | Latency (ms): p50=\(String(format: "%.3f", p50)), p95=\(String(format: "%.3f", p95))"
        )
    }
}

// MARK: - Main Execution

benchmarkVectorMathAccelerators()
benchmarkVectorStorageMemoryAndLeaks()
benchmarkHNSWIndexRecallAndDeterminism()
benchmarkHNSWScaleAdvantage()
benchmarkHardNLPParameterSweeps()

print("\n=== All Benchmarks Complete ===")
