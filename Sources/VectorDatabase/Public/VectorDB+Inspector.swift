extension VectorDatabase {

    /// Provides a read-only introspection API to explore the internal HNSW graph structure.
    /// This is strictly for visualization or debugging and does not allow modifications.
    public nonisolated var inspector: Inspector {
        Inspector(db: self)
    }

    /// A lightweight, stateless viewer into the VectorDatabase's graph structure.
    public struct Inspector: Sendable {
        private let db: VectorDatabase

        internal init(db: VectorDatabase) {
            self.db = db
        }

        /// Returns the external ID of the current HNSW entry point.
        /// Returns `nil` if the graph is empty, if all nodes are tombstoned, or if operating in Flat mode.
        public func entryPointID() async -> String? {
            await db._inspectEntryPointID()
        }

        /// Returns the randomly assigned level of the given node in the HNSW graph (e.g., `0` for the base layer).
        /// Returns `nil` if the node does not exist, is tombstoned, or if operating in Flat mode.
        public func nodeLevel(of id: String) async -> Int? {
            await db._inspectNodeLevel(of: id)
        }

        /// Returns the number of layers this node participates in (from layer `0` up to `nodeLevel`).
        /// This is strictly computed as `nodeLevel(of: id) + 1`.
        /// Returns `nil` if the node does not exist, is tombstoned, or if operating in Flat mode.
        public func layerCount(of id: String) async -> Int? {
            guard let level = await nodeLevel(of: id) else { return nil }
            return level + 1
        }

        /// Returns the neighbors of a given node at a specific layer.
        ///
        /// - Returns:
        ///   - `nil`: If the node does not exist, is tombstoned, or if the `layer` is invalid (greater than `nodeLevel`), or in Flat mode.
        ///   - `[]` (Empty Array): If the node exists and the layer is valid, but it currently has no live neighbors at this layer.
        ///   - `[String]`: An array of external string IDs representing the live neighbors. Tombstoned nodes are strictly filtered out.
        public func neighbors(of id: String, atLayer layer: Int) async -> [String]? {
            await db._inspectNeighbors(of: id, atLayer: layer)
        }

        /// Bulk topology fetch for testing graph determinism across identical instances.
        /// Extracts the entire HNSW graph topology (entry points and all neighborhood edges) in a single actor hop.
        public func fullTopologySnapshot() async -> (
            entryPoint: String?, entryPointLevel: Int,
            nodes: [String: (level: Int, neighborsByLayer: [[String]])]
        ) {
            await db._inspectFullTopologySnapshot()
        }

        /// `Codable`-friendly equivalent of `fullTopologySnapshot()`, carrying the
        /// exact same data as `GraphTopology`/`GraphNode` value types instead of a
        /// nested tuple (tuples cannot conform to `Codable`). Intended for callers
        /// that want to serialize the graph — e.g. to JSON, for a UI that renders or
        /// exports the HNSW graph — rather than consume the tuple form directly.
        public func fullTopologySnapshotCodable() async -> GraphTopology {
            let snapshot = await fullTopologySnapshot()
            let nodes = snapshot.nodes.mapValues {
                GraphNode(level: $0.level, neighborsByLayer: $0.neighborsByLayer)
            }
            return GraphTopology(
                entryPoint: snapshot.entryPoint,
                entryPointLevel: snapshot.entryPointLevel,
                nodes: nodes)
        }
    }
}

/// A JSON-serializable snapshot of the HNSW graph's structure — the `Codable`
/// counterpart to the tuple returned by `Inspector.fullTopologySnapshot()`.
/// See `Inspector.fullTopologySnapshotCodable()`.
public struct GraphTopology: Codable, Sendable {
    public let entryPoint: String?
    public let entryPointLevel: Int
    public let nodes: [String: GraphNode]
}

/// A single node's position in the HNSW graph: which level it was assigned,
/// and its live neighbor IDs at each layer from `0` up to that level.
public struct GraphNode: Codable, Sendable {
    public let level: Int
    public let neighborsByLayer: [[String]]
}
