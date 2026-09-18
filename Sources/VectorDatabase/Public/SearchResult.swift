/// SearchResult.swift — Public search result type.
/// Implementation: Phase 8 (public API design).

import Foundation

/// Represents a single matched document in a vector search.
///
/// `Codable` so a client layer (e.g. a UI that saves search history, exports
/// results to JSON, or hands results to a visualization component) can
/// serialize results without hand-rolling a parallel DTO. All stored properties
/// are already Codable-compatible, so this adds no behavior change.
public struct SearchResult: Sendable, Codable {
    /// The external identifier provided during insertion.
    public let id: String
    
    /// The similarity score between the query and this vector.
    /// Higher values indicate greater similarity.
    public let score: Float
    
    /// Optional metadata associated with this vector, provided during insertion.
    public let metadata: [String: String]?
    
    public init(id: String, score: Float, metadata: [String: String]? = nil) {
        self.id = id
        self.score = score
        self.metadata = metadata
    }
}
