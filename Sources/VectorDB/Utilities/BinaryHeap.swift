/// BinaryHeap.swift — Pure-Swift min/max binary heap for HNSW priority queues (Phase 4, §8.5).
///
/// Dependency policy: zero-dependency path (see Package.swift). The package avoids
/// `swift-collections`' Heap to keep maximum portability and no supply-chain risk.
///
/// DESIGN NOTE — Comparator direction (§8.5 Pitfall):
/// `searchLayer` uses TWO heaps with OPPOSITE ordering:
///   - `MinHeap<Candidate>` — candidates to explore, ordered by ASCENDING score
///     (pop the CLOSEST candidate next, i.e. the one with the HIGHEST score in
///     the "higher = more similar" convention — so actually pop the MAX score).
///   - `MaxHeap<Candidate>` — result set, ordered by DESCENDING score
///     (peek/pop the FARTHEST candidate so we can evict it if a better one
///     arrives).
///
/// This naming follows the PAPER's original distance language:
///   "min-heap by distance" = min-heap by (– score) = max-heap by score = pop highest first
///   "max-heap by distance" = max-heap by (– score) = min-heap by score = pop lowest first
///
/// To avoid confusion, we expose the heaps with role-descriptive names in the
/// searchLayer code and provide both `popMin`/`popMax` on a single `BinaryHeap`
/// struct, but we'll use it through `MinHeap` / `MaxHeap` typealiases that are
/// actually the same type with different nominal usage.
///
/// The unit test in HeapOrderingTests MUST verify ordering with known values
/// before anything builds on top of these heaps (§8.5 note).

// MARK: - Candidate

/// A scored graph node. Higher score = more similar to query.
/// This is the element type used in both heaps inside `searchLayer`.
struct Candidate {
    let id: Int32
    let score: Float
}

// MARK: - BinaryHeap

/// Generic binary heap. `isHigherPriority` defines ordering:
///   - Pass `{ $0.score > $1.score }` for a max-heap (pop highest score first).
///   - Pass `{ $0.score < $1.score }` for a min-heap (pop lowest score first).
struct BinaryHeap<Element> {
    private var elements: [Element]
    private let isHigherPriority: (Element, Element) -> Bool

    init(isHigherPriority: @escaping (Element, Element) -> Bool) {
        self.elements = []
        self.isHigherPriority = isHigherPriority
    }

    init(_ initial: [Element], isHigherPriority: @escaping (Element, Element) -> Bool) {
        self.isHigherPriority = isHigherPriority
        self.elements = initial
        // Build-heap in O(N): sift down from middle toward root.
        for i in stride(from: elements.count / 2 - 1, through: 0, by: -1) {
            siftDown(from: i)
        }
    }

    var count: Int { elements.count }
    var isEmpty: Bool { elements.isEmpty }

    /// Read-only access to all elements (order is heap order, NOT sorted).
    var allElements: [Element] { elements }

    /// The highest-priority element without removing it.
    func peek() -> Element? { elements.first }

    /// Push a new element into the heap. O(log N).
    mutating func push(_ element: Element) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }

    /// Remove and return the highest-priority element. O(log N).
    @discardableResult
    mutating func pop() -> Element? {
        guard !elements.isEmpty else { return nil }
        elements.swapAt(0, elements.count - 1)
        let top = elements.removeLast()
        if !elements.isEmpty { siftDown(from: 0) }
        return top
    }

    /// Return all elements sorted by priority, highest first. O(N log N).
    func sortedDescending() -> [Element] {
        var copy = self
        var result: [Element] = []
        result.reserveCapacity(elements.count)
        while let e = copy.pop() { result.append(e) }
        return result
    }

    // MARK: - Private heap operations

    private mutating func siftUp(from index: Int) {
        var child = index
        while child > 0 {
            let parent = (child - 1) / 2
            if isHigherPriority(elements[child], elements[parent]) {
                elements.swapAt(child, parent)
                child = parent
            } else {
                break
            }
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let left  = 2 * parent + 1
            let right = 2 * parent + 2
            var highest = parent

            if left  < elements.count && isHigherPriority(elements[left],  elements[highest]) { highest = left  }
            if right < elements.count && isHigherPriority(elements[right], elements[highest]) { highest = right }

            if highest == parent { break }
            elements.swapAt(parent, highest)
            parent = highest
        }
    }
}

// MARK: - Typed heap aliases for searchLayer

/// Candidates still to explore. Pop the CLOSEST (HIGHEST score) next.
/// In "higher = more similar" convention this is a max-heap by score.
/// Called "min-heap" in the paper (which uses distance, not similarity).
typealias CandidateMinHeap = BinaryHeap<Candidate>  // pop highest score first

/// Current best result set. Peek/pop the FARTHEST (LOWEST score) so it can
/// be evicted if a better candidate arrives.
/// Called "max-heap" in the paper (which uses distance, not similarity).
typealias CandidateMaxHeap = BinaryHeap<Candidate>  // pop lowest score first

extension BinaryHeap where Element == Candidate {
    /// Convenience: create a heap that pops the HIGHEST score first.
    static func maxByScore(_ initial: [Candidate] = []) -> BinaryHeap<Candidate> {
        BinaryHeap<Candidate>(initial, isHigherPriority: { $0.score > $1.score })
    }

    /// Convenience: create a heap that pops the LOWEST score first.
    static func minByScore(_ initial: [Candidate] = []) -> BinaryHeap<Candidate> {
        BinaryHeap<Candidate>(initial, isHigherPriority: { $0.score < $1.score })
    }
}
