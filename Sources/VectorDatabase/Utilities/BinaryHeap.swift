/// BinaryHeap.swift — Pure-Swift min/max binary heap for HNSW priority queues (§8.5).
///
/// Dependency policy: zero-dependency path (see Package.swift). The package avoids
/// `swift-collections`' Heap to keep maximum portability and no supply-chain risk.
///
/// DESIGN NOTE — Comparator direction (§8.5 Pitfall):
/// `searchLayer` uses TWO heaps with OPPOSITE ordering:
///   - `maxByScore()` — candidates still to explore. Pops the HIGHEST score
///     (= closest) next. The paper calls this a "min-heap by distance".
///   - `minByScore()` — the current result set. Peeks/pops the LOWEST score
///     (= farthest) so it can be evicted when a better candidate arrives. The
///     paper calls this a "max-heap by distance".
///
/// The paper's naming is inverted relative to this codebase because the paper
/// works in distance space and this codebase works in "higher score = more
/// similar" space. `HNSWCorrectnessTests` verifies the ordering with known
/// values before anything builds on top of these heaps.

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

    // FIX (coverage pass): a second initializer, `init(isHigherPriority:)`,
    // used to live here — a plain-empty-heap variant taking no `initial`
    // array. It had zero call sites: `maxByScore()`/`minByScore()`, the only
    // two ways anything in this package constructs a `BinaryHeap`, both
    // route through the `initial:` initializer below with its `= []`
    // default, which already handles the empty case correctly (the
    // build-heap loop's `stride` is simply empty). Removed rather than
    // tested, since a test would only exist to cover code nothing calls.

    init(_ initial: [Element], isHigherPriority: @escaping (Element, Element) -> Bool) {
        self.isHigherPriority = isHigherPriority
        self.elements = initial
        // Build-heap in O(N): sift down from middle toward root.
        for i in stride(from: elements.count / 2 - 1, through: 0, by: -1) {
            siftDown(from: i)
        }
    }

    var count: Int { elements.count }

    // FIX (coverage pass): `var isEmpty: Bool { elements.isEmpty }` used to
    // live here. Every emptiness check in this package — inside this file's
    // own `pop()`/`siftDown` and in `HNSWIndex.searchLayer`'s entry-point
    // guard — is written against `elements.isEmpty` or another plain array's
    // `.isEmpty` directly, never against a `BinaryHeap` instance. Zero call
    // sites, confirmed by grep; removed rather than tested for the same
    // reason as the initializer above.

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

    /// Returns every element in the heap's own priority order — i.e. the exact
    /// sequence repeated `pop()` calls would produce — without mutating `self`.
    /// O(N log N).
    ///
    /// FIX S4 — RENAMED FROM `sortedDescending()`:
    /// The old name was wrong half the time. On a `minByScore()` heap the
    /// "highest priority" element is the LOWEST score, so `sortedDescending()`
    /// actually returned an ASCENDING sequence — which is why both call sites
    /// had to chain a `.reversed()` onto it to get what the name already
    /// promised. The name now describes what the method does (drain in priority
    /// order) instead of claiming an ordering it cannot know.
    func drainedInPriorityOrder() -> [Element] {
        var copy = self
        var result: [Element] = []
        result.reserveCapacity(elements.count)
        while let e = copy.pop() { result.append(e) }
        return result
    }

    /// PERFORMANCE FIX (P4): Same as `drainedInPriorityOrder()` but returns
    /// elements in REVERSED priority order (i.e. highest-priority last).
    /// Uses in-place `.reverse()` to avoid the extra array allocation that
    /// `.reversed()` creates.
    ///
    /// Used by `searchLayer` where `foundHeap` is a minByScore heap and we
    /// need best-first (highest score first) ordering.
    func drainedInPriorityOrderReversed() -> [Element] {
        var copy = self
        var result: [Element] = []
        result.reserveCapacity(elements.count)
        while let e = copy.pop() { result.append(e) }
        result.reverse()  // in-place, no extra allocation
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
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var highest = parent

            if left < elements.count && isHigherPriority(elements[left], elements[highest]) {
                highest = left
            }
            if right < elements.count && isHigherPriority(elements[right], elements[highest]) {
                highest = right
            }

            if highest == parent { break }
            elements.swapAt(parent, highest)
            parent = highest
        }
    }
}

// MARK: - Candidate heap factories

extension BinaryHeap where Element == Candidate {
    /// Creates a heap that pops the HIGHEST score first.
    /// Used for the "candidates still to explore" queue in `searchLayer`.
    static func maxByScore(_ initial: [Candidate] = []) -> BinaryHeap<Candidate> {
        BinaryHeap<Candidate>(initial, isHigherPriority: { $0.score > $1.score })
    }

    /// Creates a heap that pops the LOWEST score first.
    /// Used for the bounded result set in `searchLayer` and the top-k set in
    /// `FlatIndex.search`, where the worst member must be cheap to evict.
    static func minByScore(_ initial: [Candidate] = []) -> BinaryHeap<Candidate> {
        BinaryHeap<Candidate>(initial, isHigherPriority: { $0.score < $1.score })
    }
}
