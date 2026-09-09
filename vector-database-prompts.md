# SwiftVectorDB — Step-by-Step Build Prompts

Derived strictly from `vector-database-documentation.md`. Every phase, every
sub-section, and the edge-case checklist each get their own prompt — nothing in the
guide is skipped. Sections §17 (Glossary) and §18 (References) are reference material
used *inside* the other prompts, not standalone build steps, so they don't get a
dedicated prompt of their own.

## How to use this

1. Attach or paste `vector-database-documentation.md` into your coding
   assistant's context at the start of the session (and re-attach it in any new
   session/thread) — every prompt below assumes the assistant can see it.
2. Send **Prompt 0** once, at the very start of the project.
3. Send the remaining prompts **one at a time, in order**. Don't start prompt *N+1*
   until prompt *N*'s "Definition of Done" is actually green.
4. If the assistant proposes anything not backed by the guide, it should say so and
   ask you before proceeding — that's Prompt 0 doing its job.

---

## Prompt 0 — Ground rules (send once, first)

```
You are helping me build the project described in the attached file
`vector-database-documentation.md` ("the guide"). Read it in full before
doing anything else.

Rules for this entire project, across every future message I send you:
1. The guide is the single source of truth. Do not introduce features, APIs,
   dependencies, file layouts, or design decisions that aren't in it.
2. If something I ask for seems to conflict with the guide, or the guide is
   ambiguous about it, stop and ask me instead of guessing.
3. Follow the guide's own structure: each phase has Goal, Concepts, Design
   Decisions, Implementation, Code Skeleton, Pitfalls, and Definition of Done.
   When you implement a phase, explicitly work through all of these, not just
   the code skeleton.
4. Do not skip Phase 1 (Flat Index) or reorder phases — the guide is explicit
   that Phase 1 becomes the permanent correctness oracle for everything after it.
5. At the end of every phase, list the phase's "Definition of Done" checklist
   items from the guide and tell me, for each one, whether it's satisfied and
   how you verified it (test run, benchmark output, etc.) — don't just claim
   "done".

Confirm you've read the guide and summarize its 11 build phases back to me in
one line each, so I can confirm you have the right mental model before we start.
```

---

## Prompt 1 — Vision, non-goals, and environment (§1–§2)

```
Using only §1 (Vision & Non-Goals) and §2 (Environment & Project Setup) of the
guide:

1. Restate the 5-line usage example from §1 and the explicit non-goals list —
   confirm you understand what is deliberately OUT of scope for v1
   (distributed operation, arbitrary metadata query language, non-Apple
   platforms, GPU/Metal compute, exact search at scale).
2. Bootstrap the package exactly as shown in §2: run
   `swift package init --type library --name VectorDB` inside a
   `SwiftVectorDB` folder, then write `Package.swift` to match the guide's
   snippet (iOS 15 / macOS 12 platforms, the VectorDB library product, the
   commented-out swift-collections dependency, `-Ounchecked` gated to release
   config only).
3. Make an explicit call on the dependency policy decision described in §2
   (pure-Swift zero-dependency binary heap vs. swift-collections' `Heap`) and
   tell me which one we're using before Phase 4 starts — don't leave this
   undecided.
4. Do NOT enable `-Ounchecked` behavior beyond what's in the Package.swift
   snippet, and do not touch it further until Phase 1's tests are green, per
   the guide's explicit warning.

Definition of Done for this step: package builds empty, targets exist,
dependency-policy decision is recorded in a comment in Package.swift.
```

---

## Prompt 2 — Project structure & architecture (§3–§4)

```
Using only §3 (Project Structure) and §4 (Architecture Overview) of the guide:

1. Scaffold the exact folder/file tree shown in §3 under Sources/VectorDB and
   Tests/VectorDBTests and Benchmarks/VectorDBBenchmarks — empty files are
   fine for now, we'll fill them phase by phase. Note that IDMap.swift is
   documented in §3 as carrying per-vector metadata as well as the id
   mapping — keep that in mind for later phases, don't design it away.
2. Write the `VectorIndex` protocol exactly as given in §3, in
   Core/IndexProtocol.swift.
3. Add the architecture diagram from §4 as a comment block at the top of
   Core/IndexProtocol.swift (or a README section) so the data-flow for
   insert() and search() described in §4 is documented in the repo itself.
4. Explain back to me, in your own words, why FlatIndex and HNSWIndex are
   kept as permanently separate implementations (§3's explanation) and why
   IndexProtocol exists — I want to confirm you understand the reasoning,
   not just copy the text.

Do not write FlatIndex or HNSWIndex bodies yet — that starts in Phase 1.
```

---

## Prompt 3 — Phase 1: Flat Index & brute-force search (§5)

```
Implement Phase 1 of the guide (§5) end to end:

1. Create `DistanceMetric` exactly as specified.
2. Create the naive scalar `VectorMath.similarity(...)` implementation as
   given in §5's corrected Implementation section (no Accelerate, no unsafe
   pointers yet — that's intentional and only changes in Phase 2). Use the
   exact "higher score = more similar" convention described, including the
   sign-flip for Euclidean distance.
3. Implement `FlatIndex` conforming to `VectorIndex`, matching the guide's
   skeleton (row-major flattened storage, `idToSlot`/`slotToID` bookkeeping).
   For `remove`, follow the guide's explicit note: a naive "mark and filter"
   is acceptable for the MVP; the real swap-remove implementation waits for
   Phase 3's VectorStorage.
4. Address both Pitfalls listed in §5: sort-direction correctness across all
   three metrics, and empty-index / `k=0` / `k > count` handling.
5. Write `FlatIndexTests` covering exactly the Definition of Done in §5:
   insert 10k random vectors and manually verify top-k against a
   numpy-equivalent reference computation for at least 5 queries; empty-index
   and k=0 edge cases; all three metrics.

Report the Definition of Done checklist from §5, item by item, with evidence.
```

---

## Prompt 4 — Phase 2: SIMD acceleration with Accelerate (§6)

```
Implement Phase 2 of the guide (§6):

1. Replace the *body* of the `VectorMath.similarity` function from Phase 1
   with the Accelerate/vDSP implementation shown in §6 — keep the exact same
   function signature so FlatIndex needs no changes, per the guide's
   explicit note on this.
2. Implement `dot`, `normalize`, and `batchDot` exactly as specified,
   including the zero-vector guard in `normalize` (`sumSq > 1e-12`).
3. Implement the cosine-via-pre-normalization strategy described: normalize
   vectors once at insert time so cosine similarity degenerates to a plain
   dot product at query time.
4. Address all four Pitfalls in §6: zero vectors, NaN/Inf inputs, memory
   alignment (confirm you're not hand-rolling alignment logic), and
   dimension-mismatch validation happening before unsafe pointer code is
   reached.
5. Meet the Definition of Done in §6: benchmark vDSP dot product vs. the old
   naive loop on 384-dim vectors over 100k iterations and report the actual
   speedup number; confirm cosine-via-normalization matches the textbook
   dot/(normA*normB) formula within float epsilon; unit-test zero-vector
   insert rejection/handling.

Report the §6 Definition of Done checklist with evidence (actual benchmark
numbers, not estimates).
```

---

## Prompt 5 — Phase 3: unsafe memory & contiguous storage (§7)

```
Implement Phase 3 of the guide (§7):

1. Implement `VectorStorage` exactly as specified: a `final class` (not a
   struct — the guide explains why), one contiguous
   `UnsafeMutablePointer<Float>` buffer, row-major layout, geometric-doubling
   growth, `deinit` freeing memory.
2. Implement `append`, `pointer(toSlot:)`, `mutablePointer(toSlot:)`, and the
   private `grow()` exactly as given.
3. Read the "Pitfalls — read this section twice" subsection in §7 and, before
   writing any calling code elsewhere, tell me in your own words: (a) why
   caching a pointer from `pointer(toSlot:)` across an `append` call is
   dangerous, (b) why `VectorStorage` must be a class and never a struct
   without full copy-on-write, and (c) why only slot indices — never raw
   pointers — should be stored in longer-lived structures like HNSWNode.
4. Now go back and update `FlatIndex.remove` to the real swap-remove
   implementation the guide defers to this phase, backed by VectorStorage.
5. Meet the Definition of Done in §7: insert 1M vectors of dimension 384 with
   no leak (Instruments Leaks/Allocations); a stress test interleaving
   append and reads on one thread verifying growth doesn't corrupt existing
   data (byte-for-byte against a reference `[[Float]]`); and the specific
   "dangling pointer across grow()" regression test described (capture a
   pointer, force a resize via 10 more inserts, verify a fresh
   `pointer(toSlot:)` call returns correct data).

Report the §7 Definition of Done checklist with evidence.
```

---

## Prompt 6 — Phase 4a: HNSW parameters and data structures (§8.1–§8.3)

```
Implement only §8.1–§8.3 of the guide (concepts, parameters, and data
structures) — no insert/search logic yet:

1. Implement `HNSWParameters` exactly as given, including the derived
   `Mmax0 = M * 2` and `mL = 1 / ln(M)` defaults, and the on-device
   recommendation (M=16, efConstruction=100, efSearch=64) as the documented
   default rationale.
2. Implement `HNSWNode` exactly as given.
3. Implement `GraphStorage` per the fixed-width adjacency design in §8.3:
   layer 0 as one contiguous `UnsafeMutablePointer<Int32>` sized
   `capacity * Mmax0`, padded with `emptySlot = -1`; per-layer sparse storage
   for upper layers, allocated lazily the first time a node reaches that
   level (as described, not pre-allocated for every node at every layer).
4. Explain back to me the design tradeoff called out in §8.3: why fixed-width
   adjacency is deliberately chosen over a simpler `[[Int32]]` list, given
   that this data will later live in an mmap'd file (Phase 6).

Do not implement `insert`, `search`, or `searchLayer` in this step — those
are Prompts 7–9.
```

---

## Prompt 7 — Phase 4b: level assignment and the search-layer primitive (§8.4–§8.5)

```
Implement §8.4 and §8.5 of the guide:

1. Implement `randomLevel()` exactly as given (Malkov & Yashunin exponential
   decay). Use a seeded PRNG you control (or a seeded
   SystemRandomNumberGenerator), per the guide's explicit instruction — do
   NOT use the default global `Double.random`, because deterministic graph
   construction is required for reproducible recall regression tests later.
2. Implement `searchLayer` exactly as given: min-heap of candidates
   (closest-first) and max-heap of found results (farthest-first), using the
   guide's "higher score = more similar" convention throughout.
3. Before moving on, write the standalone unit test the guide explicitly
   calls for in the note under §8.5: directly test heap ordering against
   known values, independent of the rest of HNSW, to catch the
   distance-minimization-vs-similarity-maximization comparator bug the guide
   warns about.
4. Confirm — and show me a test proving it — that the same seed plus the
   same insert order produces a byte-identical graph structure across two
   separate runs (this is required later for §8's Definition of Done, but
   verify the RNG determinism now while it's cheap to isolate).
```

---

## Prompt 8 — Phase 4c: HNSW insert and neighbor selection (§8.6)

```
Implement §8.6 of the guide:

1. Implement `insert(internalID:vector:)` exactly as given: the
   first-node/entry-point bootstrap case, Phase A (greedy ef=1 descent from
   the top layer down to just above the new node's level), and Phase B
   (full efConstruction search + neighbor selection + bidirectional wiring +
   pruning from min(entryPointLevel, level) down to 0).
2. Implement `selectNeighborsHeuristic` exactly as given — the
   diversity-favoring heuristic, not the naive top-m closest — including the
   backfill step when diversity pruning leaves the selection short of `m`.
3. Address the two insert-specific Pitfalls from §8's Pitfalls subsection
   that apply here: entry point level staying correct as new nodes are
   inserted, and the recursive-lock-reentrancy concern (pruning other nodes'
   neighbor lists from inside an in-progress insert) — flag explicitly how
   your implementation avoids reacquiring a non-reentrant lock, even though
   Phase 5's actual locking layer doesn't exist yet; note it as a TODO
   comment referencing Phase 5 if needed.

Don't implement query/search yet — that's the next prompt.
```

---

## Prompt 9 — Phase 4d: HNSW query search, and Phase 4 sign-off (§8.7 + §8 Pitfalls/DoD)

```
Implement §8.7 and close out Phase 4:

1. Implement `search(query:k:ef:)` exactly as given in §8.7 (descend the
   upper layers with ef=1 greedy search, then run the final efSearch beam
   search at layer 0).
2. Go through every remaining Pitfall listed at the end of §8 that Prompts
   6–8 haven't already covered: disconnected-graph-after-deletion (note this
   is fully handled in Phase 7, just confirm your data structures don't
   block that later work), and the small-collection overhead concern —
   implement the FlatIndex/HNSWIndex threshold switch described (e.g. below
   ~2,000 vectors use FlatIndex transparently) inside whatever will become
   the public-facing layer, or flag it clearly as deferred to Phase 8 if
   that's a cleaner seam.
3. Meet the full Definition of Done for §8: recall@10 ≥ 0.95 against the
   FlatIndex oracle on a 50k-vector synthetic dataset at the chosen default
   efSearch; record actual wall-clock insert time for 50k vectors on a real
   device as your regression baseline; confirm deterministic graph
   construction (same seed + same insert order ⇒ byte-identical graph)
   end-to-end, not just for the primitives tested in Prompt 7.

Report the full §8 Definition of Done checklist with evidence, including the
actual recall number and actual timing number — not placeholders.
```

---

## Prompt 10 — Phase 5: concurrency & thread safety (§9)

```
Implement Phase 5 of the guide (§9):

1. Implement `ReadWriteLock` exactly as given, wrapping `pthread_rwlock_t`.
2. Implement the two-layer design described: a plain (non-actor) `Engine`
   class internally using `withRead`/`withWrite` around the real
   `VectorStorage`/`GraphStorage`/index work, wrapped by the public
   `actor VectorDB` for Swift Concurrency ergonomics. Implement enough of
   `VectorDB.search` and `VectorDB.insert` to demonstrate this pattern (the
   full public API surface is Phase 8 — don't build it out fully here).
3. Explain back to me, in your own words, why the guide deliberately avoids
   making `Engine` itself an actor (the reasoning given right after the code
   sample in §9) — I want to confirm you understand it's intentional, not an
   oversight to "fix" later.
4. Address all three Pitfalls in §9: writer starvation on Darwin's
   non-fair rwlock, never holding the rwlock during slow disk I/O (only
   around the in-memory handoff), and ensuring `grow()` on VectorStorage/
   GraphStorage only ever happens under the write lock.
5. Meet the Definition of Done in §9: an 8-reader / 1-writer stress test run
   for 60 seconds under Thread Sanitizer with zero races reported, and a
   comparison confirming search latency under concurrent load degrades
   reasonably rather than collapsing under lock contention.

Report the §9 Definition of Done checklist with evidence (actual TSan run
output, actual latency numbers).
```

---

## Prompt 11 — Phase 6a: persistence file format design (§10.1–§10.2)

```
Implement §10.1 and §10.2 of the guide — design only, minimal code:

1. Explain back to me, using the guide's own numbers, why mmap is used
   instead of reading the whole file into memory (the 100k-vector /
   ~146MB example from §10.1).
2. Implement `FileFormat.swift` with the exact fixed-size header layout from
   §10.2: magic bytes, formatVersion, dimension, vectorCount, capacity,
   metric byte, hnswM/hnswMmax0, entryPointID/entryPointLevel, the three
   section offsets, and the checksum field. For the checksum, pick one
   concrete algorithm per the corrected guide's guidance (a fast
   non-cryptographic hash such as CRC32C or 64-bit FNV/xxHash — it only
   needs to catch truncation/corruption, not resist tampering) and document
   your choice in a comment.
3. Lay out the vector section, layer-0 graph section, sparse upper-layer
   graph section, and ID-map section (including the metadata blob) exactly
   as described — the ID map section is explicitly NOT mmap-addressed
   directly per the guide, since it's small; implement it as a fully-loaded
   in-memory structure on load.
4. Do not implement `MappedFile` or the WAL yet — those are the next two
   prompts.
```

---

## Prompt 12 — Phase 6b: the mmap wrapper (§10.3)

```
Implement §10.3 of the guide:

1. Implement `MappedFile` exactly as given: open/create with the right
   flags, `fstat`/`ftruncate` to size, `mmap` with `PROT_READ|PROT_WRITE` and
   `MAP_SHARED`, `resize(to:)` implementing the unmap→ftruncate→remap
   sequence, `sync()` via `msync(..., MS_SYNC)`, and `deinit` unmapping and
   closing the fd.
2. Address the critical Darwin-specific gotcha called out right after the
   code sample: growing a mapping can hand back a different base address, so
   `VectorStorage`/`GraphStorage`, when backed by a `MappedFile`, must always
   recompute their base pointer from `mappedFile.pointer` + a fixed offset
   on each access rather than caching their own pointer. Show me exactly
   where in your VectorStorage/GraphStorage code (from Phases 3 and 4) this
   change is applied when they're backed by a MappedFile.
3. Do not implement the WAL or PersistenceManager yet — that's the next
   prompt.
```

---

## Prompt 13 — Phase 6c: WAL, snapshot/compaction, and Phase 6 sign-off (§10.4 + §10 Pitfalls/DoD)

```
Implement §10.4 of the guide and close out Phase 6:

1. Implement the WAL exactly as described: append-only file of
   `(opcode: insert/delete, internalID, vector bytes, timestamp)` records,
   with periodic (configurable) `fsync`, using `WALOpcode` as given.
2. Implement `PersistenceManager.save()` and `.load()` exactly as given:
   save = write to a `.tmp` file, atomic `replaceItemAt` rename over the
   real file, then truncate the WAL; load = read the last good snapshot via
   mmap, then replay any WAL records written after it.
3. Address every Pitfall listed at the end of §10: checksum validation
   before trusting a snapshot on load, refusing files from a newer
   formatVersion with `.unsupportedFileVersion`, propagating typed errors
   (never silent no-ops) on disk-full/permission-denied, wrapping save() in
   a background-task extension for iOS backgrounding, and explicitly noting
   (as a documented v2 concern, not solving it now) that cross-process
   access via app extensions needs real file locking beyond this design.
4. Meet the full Definition of Done for §10: simulate a crash mid-insert-
   burst and verify recovery loses at most the configured WAL flush
   interval with no corruption; simulate a crash mid-save() (inject a delay
   + SIGKILL before rename completes) and verify the old snapshot is still
   intact and loadable; load a deliberately truncated/corrupted file and
   verify a typed error, not a crash; measure actual resident memory via
   Instruments before/after switching to mmap on a 100k-vector store and
   report the real numbers.

Report the full §10 Definition of Done checklist with evidence.
```

---

## Prompt 14 — Phase 7: deletion & updates (§11)

```
Implement Phase 7 of the guide (§11):

1. Implement tombstone-based `remove(internalID:)` and the tombstone-aware
   `search` filtering exactly as given (fetch `k + tombstoned.count`, then
   filter, then take `k`).
2. Implement the rebuild trigger: track `tombstoned.count / nodes.count` and
   trigger a full rebuild once it crosses the guide's suggested 10–20%
   threshold, re-inserting live vectors into a fresh HNSWIndex, then
   atomically swapping it in under the write lock, off the main actor.
3. Implement `update` as tombstone-old + insert-fresh, exactly as the guide
   specifies — do not attempt to mutate an existing node's vector/edges in
   place.
4. Address all three Pitfalls in §11: exposing `liveCount`/`tombstonedCount`
   via a stats surface so tombstone bloat is visible to the app developer;
   correct entry-point reassignment when the current entry point gets
   tombstoned (pick a new one from the highest surviving non-tombstoned
   level, without removing the old node from the graph structure itself);
   and scheduling the rebuild at an appropriate time (e.g. during save() or
   backgrounded/charging) rather than synchronously inside a user-facing
   delete() call.
5. Meet the Definition of Done in §11: delete 30% of a 20k-vector index and
   verify search never returns a tombstoned ID; trigger a rebuild and verify
   recall@10 against the FlatIndex oracle recovers to baseline (not just
   stays fine, since tombstoning alone won't regress it — the real test is
   rebuild doesn't regress it); and a test that specifically tombstones the
   current entry point, then performs further inserts/searches successfully.

Report the §11 Definition of Done checklist with evidence.
```

---

## Prompt 15 — Phase 8: public API design (§12)

```
Implement Phase 8 of the guide (§12), finalizing the public surface:

1. Implement the full `public actor VectorDB` API exactly as given: init,
   insert, batchInsert, search, delete, update, save, close, stats(), and
   the Stats struct.
2. Implement `SearchResult` and the full `VectorDBError` enum exactly as
   given.
3. Make every "Decisions to make explicit in your API docs" item from §12 an
   actual documented decision (in doc comments on the public API, not just
   in your head): duplicate-ID policy (insert throws `.duplicateID`, update
   is the explicit upsert path); `path: nil` giving a fully-functional
   in-memory-only mode with no forced disk I/O; the synchronous vs. async
   reasoning for insert/search vs. save(); the close() semantics you defined
   back in the corrected guide's Phase 8 notes (flush WAL, unmap/close file
   descriptors, idempotent to call more than once); and batchInsert's
   single-lock-for-the-whole-batch behavior plus its documented
   all-or-nothing vs. partial-failure policy.
4. Wire this public actor to actually use FlatIndex below the small-
   collection threshold and HNSWIndex above it (from Phase 4's deferred
   item), transparently, with no public API difference.

This prompt is about assembling and finalizing the public surface from all
previous phases — it should not require new core algorithm work.
```

---

## Prompt 16 — Phase 9: testing strategy (§13)

```
Implement Phase 9 of the guide (§13) as a complete test suite, building on
everything from Prompts 3–15:

1. Ensure `FlatIndexTests`, `HNSWCorrectnessTests`, `PersistenceTests`,
   `ConcurrencyTests`, and `EdgeCaseTests` all exist as named in §3/§13 (some
   were started in earlier prompts — consolidate and complete them here).
2. Implement `TestFixtures.swift` with a seeded deterministic random vector
   generator, used by every other test file, per §13.
3. Implement `HNSWCorrectnessTests` exactly as described: for dataset sizes
   1k / 10k / 50k, build both a FlatIndex and HNSWIndex from the same
   vectors, run identical queries against both, and assert
   recall@k ≥ threshold averaged over many queries (not a single query).
4. Implement the property-based-style fuzzing test described at the end of
   §13: random sequences of insert/delete/search against a FlatIndex-backed
   reference and the real HNSWIndex, asserting consistency (within recall
   tolerance) after arbitrary operation interleavings.
5. Cross-check: every EdgeCaseTests entry should map to a real row in §16's
   table — don't finish this prompt until that mapping is complete (the next
   prompt verifies this explicitly, but do the work now).
```

---

## Prompt 17 — Edge case & failure mode checklist walkthrough (§16)

```
Go through the Edge Case & Failure Mode Checklist table in §16 of the guide
row by row — every single row, in table order (Input validation, Insert,
Search, Delete, Persistence, Concurrency, Memory, Scale categories).

For each row:
1. Quote the Case and Expected Behavior from the table.
2. Tell me whether it's already covered by an existing test from earlier
   phases (name the test), or needs a new one.
3. If it needs a new test, write it now in EdgeCaseTests.swift (or the most
   appropriate existing test file if the guide implies otherwise, e.g. the
   Int32-overflow-at-4096-dim case may belong alongside VectorStorage/
   GraphStorage tests).

At the end, give me a table mirroring §16's exactly, with a third column you
add: "Test name" — so I can see every row in the guide is now backed by an
actual test, not just a mental note, per the guide's own instruction at the
top of §16.
```

---

## Prompt 18 — Phase 10: benchmarking & tuning (§14)

```
Implement Phase 10 of the guide (§14):

1. Build the standalone executable target `Benchmarks/VectorDBBenchmarks/
   main.swift` — explicitly NOT using XCTest's `measure {}`, per the guide's
   instruction that its statistical reporting is too thin for this.
2. Implement all four things §14 asks the benchmark suite to do: load/
   generate a dataset (synthetic random unit vectors first, then real
   CoreML sentence-embedding output before shipping, per the guide's note
   that clustering behavior differs); sweep efSearch across
   10/25/50/100/200 and produce the recall-vs-latency curve; measure build
   time, p50/p95/p99 query latency, memory footprint (RSS via
   mach_task_basic_info), and energy impact (Instruments' Energy Log
   template on a real device, not the simulator); and compare HNSW against
   the FlatIndex baseline at the same dataset size.
3. Create the checked-in `BENCHMARKS.md` described at the end of §14 and
   populate it with your actual results — this is meant to make regressions
   visible in code review, so it needs real numbers, not placeholders.
4. Use the recall-vs-latency curve you produced to make and document a final
   decision on the shipped default `efSearch`, per §8.2's note that this
   should come from real data, not a guess.
```

---

## Prompt 19 — Phase 11: packaging, CI, and distribution (§15)

```
Implement Phase 11 of the guide (§15):

1. Set up CI (GitHub Actions or equivalent) running `swift test` on a Mac
   runner, plus a build-only job using
   `xcodebuild -scheme VectorDB -destination 'platform=iOS Simulator,name=iPhone 15'`
   to catch iOS-specific compile issues, exactly as specified.
2. Add a dedicated CI job running the concurrency test suite under
   `-sanitize=thread`, separate from the regular test job, per the guide's
   warning that this must not be allowed to regress silently.
3. Document the semantic versioning policy described: bump the major version
   on any on-disk format change, paired with the `formatVersion` field from
   §10.2, so an app updating the SDK never silently corrupts users' on-disk
   stores.
4. Write the README leading with the 5-line usage example from §1, then
   linking into the full guide (or a trimmed public version of it) for
   internals, as specified.
```

---

## Prompt 20 — Final milestone verification (§19) ##

```
Go through the Milestone Checklist in §19 of the guide, M0 through M11, in
order. For each milestone:

1. Quote the milestone's exact criteria from §19.
2. Tell me, with concrete evidence (test names, benchmark numbers, CI job
   links/output) gathered across all previous prompts, whether it's actually
   met — not just "should be, we did the phase".
3. If any milestone isn't fully met, tell me specifically which prior prompt
   needs revisiting before we can consider the project's v1 scope (as
   defined in §1's non-goals) complete.

Give me the final checklist back with each box explicitly marked done or not
done, mirroring §19's own checkbox format.
```
