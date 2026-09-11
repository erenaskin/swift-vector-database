extension VectorDB {
    
    /// Provides a read-only introspection API to explore the internal HNSW graph structure.
    /// This is strictly for visualization or debugging and does not allow modifications.
    public nonisolated var inspector: Inspector {
        Inspector(db: self)
    }
    
    /// A lightweight, stateless viewer into the VectorDB's graph structure.
    public struct Inspector: Sendable {
        private let db: VectorDB
        
        internal init(db: VectorDB) {
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
    }
    
}
