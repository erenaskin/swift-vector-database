/// DistanceMetric.swift — Public distance metric enum.
/// Introduced in Phase 1; used unchanged through all later phases.
///
/// Score convention (enforced at the VectorMath layer so no caller ever
/// branches on metric type for sort direction):
///   - .cosine / .dotProduct  → higher dot product = more similar
///   - .euclidean             → score = -distanceSquared; higher = closer
public enum DistanceMetric: Sendable, CaseIterable {
    case cosine
    case euclidean
    case dotProduct
}
