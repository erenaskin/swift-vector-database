/// HNSWParameters.swift — Tunable hyperparameters for the HNSW index (§8.2).
///
/// These parameters dominate the recall / speed / memory tradeoff for the graph.
/// Default values (M=16, efConstruction=100, efSearch=50) are chosen for on-device
/// text-embedding workloads (hundreds to low tens of thousands of vectors) favoring
/// build/insert speed over marginal recall gains at web-scale.
import Foundation

public struct HNSWParameters: Sendable {
    /// Max neighbors per node at layer 1 and above.
    /// Higher = better recall, more memory, slower insert.
    public var M: Int
    
    /// Max neighbors per node at layer 0 (derived as M * 2).
    /// Layer 0 needs denser connectivity since it's the final search layer.
    public var Mmax0: Int
    
    /// Candidate list size during insertion.
    /// Higher = better graph quality, much slower build.
    public var efConstruction: Int
    
    /// Candidate list size during query.
    /// Higher = better recall, slower query. Runtime-tunable knob.
    public var efSearch: Int
    
    /// Level-generation normalization factor (derived as 1 / ln(M)).
    /// Controls how many nodes end up on higher layers.
    public var mL: Double
    
    /// Seed for random level assignment (deterministic graph construction).
    public var seed: UInt64
    
    public init(M: Int = 16, efConstruction: Int = 100, efSearch: Int = 50, seed: UInt64 = 0x5EED) {
        self.M = M
        self.Mmax0 = M * 2
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.mL = 1.0 / log(Double(M))
        self.seed = seed
    }
    
    /// On-device recommendation defaults.
    public static let `default` = HNSWParameters()
}
