/// VectorStorage.swift — Contiguous unsafe Float buffer (row-major, N × D).
///
/// Phase 3 replacement for FlatIndex's `[Float]` storage. Designed to:
///   - Grow geometrically (O(1) amortised append, no O(N²) realloc)
///   - Be memory-mapped in Phase 6 (fixed layout, addressable by arithmetic)
///   - Support zero-overhead row-major access via raw pointer arithmetic
///
/// PITFALLS — READ THIS SECTION TWICE (§7):
///
///   (1) Use-after-grow dangling pointer:
///       `grow()` allocates a NEW buffer at a DIFFERENT address, copies all
///       existing data, then DEALLOCATES the old buffer. Any UnsafePointer
///       obtained from `pointer(toSlot:)` before a `grow()` is now a dangling
///       pointer. Rule: NEVER cache a raw pointer from this class across any
///       call that might trigger `append` (which triggers `grow`). Re-fetch
///       after every mutation.
///
///   (2) Must be a `final class`, never a struct:
///       If this were a struct, a Swift copy would duplicate the *pointer value*
///       (the address) without duplicating the underlying allocation. Both copies
///       would share the same memory and both deinits would free it → double-free.
///       A class gives reference semantics: copies share the object; deinit
///       runs exactly once.
///
///   (3) Never persist raw pointers in long-lived structures (e.g. HNSWNode):
///       Any future `append` may call `grow()` and move the buffer. Store Int32
///       slot indices everywhere outside VectorStorage, and only call
///       `pointer(toSlot:)` immediately before a math call.
///
///   (4) Thread safety: NONE. This class assumes single-threaded access.
///       Phase 5 (ReadWriteLock) wraps all access appropriately.

public final class VectorStorage {

    // MARK: - Stored properties

    /// The fixed vector dimensionality. Immutable after init.
    public let dimension: Int

    /// Number of allocated Float slots. Grows geometrically; never shrinks.
    public private(set) var capacity: Int

    /// Number of vectors written so far. Incremented by `append`, decremented
    /// by `removeLast` (used for swap-remove in FlatIndex).
    public private(set) var count: Int = 0

    /// Memory allocation backing this storage.
    public enum Allocation {
        case heap(UnsafeMutablePointer<Float>)
        case mapped(file: MappedFile, offset: Int)
    }
    
    private var allocation: Allocation

    /// The raw contiguous buffer: vector i lives at `buffer + i * dimension`.
    /// Re-calculated dynamically if backed by a mapped file to survive Darwin `mmap` resizes.
    internal var buffer: UnsafeMutablePointer<Float> {
        switch allocation {
        case .heap(let ptr):
            return ptr
        case .mapped(let file, let offset):
            // Darwin mmap gotcha: `file.pointer` may change after a resize. 
            // We MUST recompute this on every access.
            return file.pointer.advanced(by: offset).assumingMemoryBound(to: Float.self)
        }
    }

    // MARK: - Init / Deinit

    public init(dimension: Int, initialCapacity: Int = 1024) {
        precondition(dimension > 0,       "VectorStorage: dimension must be > 0")
        precondition(initialCapacity > 0, "VectorStorage: initialCapacity must be > 0")
        self.dimension = dimension
        self.capacity  = initialCapacity
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: initialCapacity * dimension)
        self.allocation = .heap(ptr)
    }

    /// Initializes a read-only or externally-managed mapped storage.
    public init(dimension: Int, capacity: Int, mappedFile: MappedFile, offset: Int, count: Int) {
        self.dimension = dimension
        self.capacity = capacity
        self.count = count
        self.allocation = .mapped(file: mappedFile, offset: offset)
    }

    deinit {
        if case .heap(let ptr) = allocation {
            ptr.deallocate()
        }
        // Mapped memory is managed by the MappedFile instance.
    }

    // MARK: - Append

    /// Appends a vector to the end of the buffer, growing if necessary.
    /// - Parameter vector: Pointer to exactly `dimension` Float values.
    /// - Returns: The slot index at which the vector was stored.
    @discardableResult
    public func append(_ vector: UnsafePointer<Float>) -> Int {
        if count == capacity { grow() }
        let dest = buffer + count * dimension
        dest.update(from: vector, count: dimension)
        let slot = count
        count += 1
        return slot
    }

    // MARK: - Access

    /// Returns a read-only pointer to the vector at `slot`.
    ///
    /// ⚠️ This pointer is ONLY valid until the next `append` call that triggers
    /// `grow()`. Never store it in a longer-lived data structure; always
    /// re-fetch after any mutation.
    public func pointer(toSlot slot: Int) -> UnsafePointer<Float> {
        precondition(slot >= 0 && slot < count,
            "VectorStorage: slot \(slot) out of bounds (count=\(count))")
        return UnsafePointer(buffer + slot * dimension)
    }

    /// Returns a read-write pointer to the vector at `slot`.
    /// Subject to the same dangling-pointer pitfall as `pointer(toSlot:)`.
    public func mutablePointer(toSlot slot: Int) -> UnsafeMutablePointer<Float> {
        precondition(slot >= 0 && slot < count,
            "VectorStorage: slot \(slot) out of bounds (count=\(count))")
        return buffer + slot * dimension
    }

    // MARK: - Swap-remove support

    /// Logically releases the last slot by decrementing `count` by 1.
    ///
    /// Used exclusively by `FlatIndex`'s swap-remove: the caller MUST have
    /// already copied the last slot's data into the target slot via
    /// `mutablePointer(toSlot:)` before calling this.
    ///
    /// The buffer is NOT zeroed — the vacated bytes are simply inaccessible
    /// via `pointer(toSlot:)` after this call (the precondition will trap).
    public func removeLast() {
        precondition(count > 0, "VectorStorage: removeLast() called on empty storage")
        count -= 1
    }

    // MARK: - Growth

    /// Doubles the buffer capacity, copies all existing data, frees the old buffer.
    ///
    /// ⚠️ After this returns, any previously obtained `UnsafePointer<Float>`
    /// into the old buffer is a DANGLING POINTER. The new buffer is at a
    /// potentially different address. This is the primary pitfall of this class.
    private func grow() {
        let newCapacity = capacity * 2
        
        switch allocation {
        case .heap(let oldPtr):
            let newBuffer = UnsafeMutablePointer<Float>.allocate(capacity: newCapacity * dimension)
            newBuffer.update(from: oldPtr, count: count * dimension)
            oldPtr.deallocate()
            self.allocation = .heap(newBuffer)
        case .mapped:
            // Task 5: Fall back to heap allocation instead of crashing.
            let newBuffer = UnsafeMutablePointer<Float>.allocate(capacity: newCapacity * dimension)
            // self.buffer automatically resolves to the correct mapped pointer
            newBuffer.update(from: self.buffer, count: count * dimension)
            // Leave the MappedFile alone (don't deallocate it since it's shared)
            self.allocation = .heap(newBuffer)
        }
        
        capacity = newCapacity
    }
}
