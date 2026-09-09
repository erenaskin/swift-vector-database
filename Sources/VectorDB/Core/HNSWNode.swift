/// HNSWNode.swift — Core node data structure for the HNSW graph (§8.3).
///
/// HNSWNode is a lightweight struct representing a single vector's presence in the graph.
/// Crucially, it does NOT store a raw pointer to its float data, because VectorStorage
/// can be reallocated (moved) at any time. Instead, it stores `vectorSlot`, which is
/// an index into VectorStorage.
struct HNSWNode: Codable {
    /// The top layer this node exists on (0...L).
    var level: Int
    
    /// The slot index into VectorStorage where the vector data is located.
    /// Used instead of raw pointers to avoid dangling pointer crashes upon growth.
    var vectorSlot: Int32
}
