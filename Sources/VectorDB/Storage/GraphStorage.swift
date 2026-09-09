/// GraphStorage.swift — Fixed-capacity adjacency storage for HNSW (§8.3).
///
/// Every node reserves `Mmax0` slots at layer 0 and `M` slots at each higher layer
/// it participates in, padded with `emptySlot`. 
///
/// DESIGN TRADEOFF (Fixed-Width vs. [[Int32]] Adjacency):
/// A dynamic `[[Int32]]` array is conceptually simpler and easier to write, but it
/// is unacceptable for a production vector database. Each inner array is a separate
/// heap allocation, leading to severe pointer-chasing and cache misses during search.
/// More importantly, when we memory-map the index to disk (Phase 6), we cannot `mmap`
/// a `[[Int32]]` array. By enforcing a fixed-width adjacency layout from the start, 
/// the entire layer is a single contiguous block of memory. Finding node X's neighbors 
/// is a guaranteed O(1) arithmetic operation (`basePointer + X * Mmax0 * 4`), which 
/// allows the OS to directly page-in exactly the bytes it needs from disk without 
/// any parsing or serialization overhead. We trade a small amount of RAM (padding 
/// with emptySlots) for massive performance and persistence simplicity.

final class GraphStorage {
    
    static let emptySlot: Int32 = -1
    
    let mMax0: Int
    let m: Int
    
    public enum Allocation {
        case heap(layer0: UnsafeMutablePointer<Int32>, upperLayers: [UnsafeMutablePointer<Int32>])
        case mapped(file: MappedFile, baseOffset: Int)
    }
    
    private var allocation: Allocation

    /// Layer 0 adjacency: single contiguous buffer of `capacity * mMax0` slots.
    /// Re-calculated dynamically if backed by a mapped file to survive Darwin `mmap` resizes.
    internal var layer0: UnsafeMutablePointer<Int32> {
        switch allocation {
        case .heap(let l0, _):
            return l0
        case .mapped(let file, let offset):
            // Darwin mmap gotcha: `file.pointer` may change after a resize. 
            // We MUST recompute this on every access.
            return file.pointer.advanced(by: offset).assumingMemoryBound(to: Int32.self)
        }
    }
    
    /// Upper layers (1...L) adjacency. Sparse per level.
    /// Re-calculated dynamically if backed by a mapped file.
    internal func upperLayerPointer(_ layer: Int) -> UnsafeMutablePointer<Int32> {
        let uIdx = layer - 1
        switch allocation {
        case .heap(_, let upperLayers):
            return upperLayers[uIdx]
        case .mapped(let file, let offset):
            // Upper layers start immediately after layer0
            let layer0Size = capacity * mMax0 * MemoryLayout<Int32>.size
            let upperLayerOffset = layer0Size + (uIdx * capacity * m * MemoryLayout<Int32>.size)
            return file.pointer.advanced(by: offset + upperLayerOffset).assumingMemoryBound(to: Int32.self)
        }
    }
    
    public private(set) var capacity: Int
    public private(set) var count: Int = 0
    
    /// Cached neighbor counts: `neighborCounts[level][internalID] -> count`.
    /// Avoiding a linear scan over `emptySlot` padding speeds up insertion.
    internal var neighborCounts: [[Int32: Int]] = []
    
    init(m: Int, mMax0: Int, initialCapacity: Int = 1024) {
        precondition(m > 0 && mMax0 > 0, "m and mMax0 must be > 0")
        precondition(initialCapacity > 0, "initialCapacity must be > 0")
        
        self.m = m
        self.mMax0 = mMax0
        self.capacity = initialCapacity
        
        let l0 = UnsafeMutablePointer<Int32>.allocate(capacity: initialCapacity * mMax0)
        l0.initialize(repeating: Self.emptySlot, count: initialCapacity * mMax0)
        self.allocation = .heap(layer0: l0, upperLayers: [])
        
        // neighborCounts for layer 0 starts empty
        self.neighborCounts.append([:])
    }
    
    /// Initializes a read-only or externally-managed mapped storage.
    init(m: Int, mMax0: Int, capacity: Int, count: Int, mappedFile: MappedFile, offset: Int, neighborCounts: [[Int32: Int]]) {
        self.m = m
        self.mMax0 = mMax0
        self.capacity = capacity
        self.count = count
        self.neighborCounts = neighborCounts
        self.allocation = .mapped(file: mappedFile, baseOffset: offset)
    }
    
    deinit {
        if case .heap(let l0, let upperLayers) = allocation {
            l0.deallocate()
            upperLayers.forEach { $0.deallocate() }
        }
        // Mapped memory is managed by the MappedFile instance.
    }
    
    // MARK: - Adjacency Access
    
    /// Returns the active neighbors for a given node at a given layer.
    func neighbors(of internalID: Int32, at layer: Int) -> [Int32] {
        guard layer < neighborCounts.count else { return [] }
        guard let count = neighborCounts[layer][internalID], count > 0 else {

            return []
        }
        
        var result: [Int32] = []
        result.reserveCapacity(count)
        
        if layer == 0 {
            let start = layer0 + Int(internalID) * mMax0
            for i in 0..<mMax0 {
                let val = start[i]
                if val != Self.emptySlot {
                    result.append(val)
                }
            }
        } else {
            let start = upperLayerPointer(layer) + Int(internalID) * m
            for i in 0..<m {
                let val = start[i]
                if val != Self.emptySlot {
                    result.append(val)
                }
            }
        }
        return result
    }
    
    /// Overwrites the full neighbor list for a given node at a given layer.
    func setNeighbors(of internalID: Int32, at layer: Int, to newNeighbors: [Int32]) {
        if layer == 0 {
            precondition(newNeighbors.count <= mMax0, "Layer 0 neighbors cannot exceed mMax0")
            let start = layer0 + Int(internalID) * mMax0
            
            for i in 0..<newNeighbors.count {
                start[i] = newNeighbors[i]
            }
            // Pad remaining with emptySlot
            for i in newNeighbors.count..<mMax0 {
                start[i] = Self.emptySlot
            }
        } else {
            precondition(newNeighbors.count <= m, "Upper layer neighbors cannot exceed m")
            
            // Ensure upper layer is allocated
            ensureUpperLayerAllocated(layer)
            let start = upperLayerPointer(layer) + Int(internalID) * m
            
            for i in 0..<newNeighbors.count {
                start[i] = newNeighbors[i]
            }
            // Pad remaining with emptySlot
            for i in newNeighbors.count..<m {
                start[i] = Self.emptySlot
            }
        }
        
        neighborCounts[layer][internalID] = newNeighbors.count
    }
    
    /// Adds a new node slot and returns its internal index.
    @discardableResult
    func addNode() -> Int32 {
        if count == capacity {
            grow()
        }
        let id = Int32(count)
        count += 1
        return id
    }

    /// Returns the number of active (non-emptySlot) neighbors for a node at a layer.
    func neighborCount(of internalID: Int32, at layer: Int) -> Int {
        guard layer < neighborCounts.count else { return 0 }
        return neighborCounts[layer][internalID] ?? 0
    }

    /// Appends a single neighbor to a node's adjacency row at a given layer,
    /// if there is room within the fixed-width limit. Used during bidirectional
    /// wiring in insert(). Caller is responsible for pruning if the limit is exceeded.
    func addNeighbor(of internalID: Int32, at layer: Int, neighborID: Int32) {
        let currentCount = neighborCount(of: internalID, at: layer)
        let limit = (layer == 0) ? mMax0 : m

        if layer == 0 {
            if currentCount < limit {
                let row = layer0 + Int(internalID) * mMax0
                row[currentCount] = neighborID
                neighborCounts[0][internalID] = currentCount + 1
            }
        } else {
            ensureUpperLayerAllocated(layer)
            if currentCount < limit {
                let row = upperLayerPointer(layer) + Int(internalID) * m
                row[currentCount] = neighborID
                neighborCounts[layer][internalID] = currentCount + 1
            }
        }
    }

    
    // MARK: - Private Growth and Allocation
    
    /// Lazily allocates an upper layer buffer the first time a node reaches it.
    private func ensureUpperLayerAllocated(_ layer: Int) {
        precondition(layer > 0, "Layer 0 is always allocated")
        
        switch allocation {
        case .heap(let l0, var upperLayers):
            var modified = false
            while upperLayers.count < layer {
                let newBuffer = UnsafeMutablePointer<Int32>.allocate(capacity: capacity * m)
                newBuffer.initialize(repeating: Self.emptySlot, count: capacity * m)
                upperLayers.append(newBuffer)
                neighborCounts.append([:])
                modified = true
            }
            if modified {
                allocation = .heap(layer0: l0, upperLayers: upperLayers)
            }
        case .mapped:
            // Mapped graph sections are pre-allocated by the PersistenceManager's file layout.
            while neighborCounts.count <= layer {
                neighborCounts.append([:])
            }
        }
    }
    
    private func grow() {
        let newCapacity = capacity * 2
        
        switch allocation {
        case .heap(let l0, let upperLayers):
            // 1. Grow layer 0
            let newLayer0 = UnsafeMutablePointer<Int32>.allocate(capacity: newCapacity * mMax0)
            newLayer0.initialize(repeating: Self.emptySlot, count: newCapacity * mMax0)
            newLayer0.update(from: l0, count: capacity * mMax0)
            l0.deallocate()
            
            // 2. Grow any existing upper layers
            var newUpperLayers: [UnsafeMutablePointer<Int32>] = []
            for oldLayer in upperLayers {
                let newLayer = UnsafeMutablePointer<Int32>.allocate(capacity: newCapacity * m)
                newLayer.initialize(repeating: Self.emptySlot, count: newCapacity * m)
                newLayer.update(from: oldLayer, count: capacity * m)
                oldLayer.deallocate()
                newUpperLayers.append(newLayer)
            }
            
            self.allocation = .heap(layer0: newLayer0, upperLayers: newUpperLayers)
            
        case .mapped:
            // A mapped file cannot be grown locally by GraphStorage because it is interleaved 
            // with VectorStorage sections. Resizing the file is the PersistenceManager's job.
            fatalError("Memory-mapped GraphStorage cannot grow independently.")
        }
        
        capacity = newCapacity
    }
}
