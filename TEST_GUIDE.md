# TEST_GUIDE.md — pure-swift-vector-database

This is the project's **single, comprehensive** testing guide. It replaces
the `test_guide.md` and `TEST.md` files — you can delete both of them (see
the note at the end of the file).

There are two main sections:
1. **Testing via terminal commands** — the scriptable path, the one CI also uses.
2. **Testing via the Xcode interface** — the path you'll use during day-to-day development.

For each command I used the format: **what it does → expected output → what
to watch out for**. I'm not just saying "run this"; I'm also explaining how
to read the output.

---

## 0. Environment Setup (REQUIRED IN EVERY TERMINAL SESSION)

```bash
cd ~/Desktop/pure-swift-vector-database   # replace with your own path
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift --version
```

**Why it's necessary:** On this machine, without `DEVELOPER_DIR` set, the
`swift` and `xcodebuild` commands either won't work or will use the wrong
toolchain. If you skip this step, most of the commands below will either
fail with nonsensical errors or silently use the wrong (old/system) Swift
version.

The `swift --version` output should show Xcode's own Swift version (e.g.
`Apple Swift version 6.x`) — not the system Swift.

---

# SECTION 1 — Testing via Terminal Commands

## 1.1 Build Health

### Command 1 — Debug build
```bash
swift build
```
**What it tests:** Whether the project compiles at a basic level. The
fastest, most frequently run command.

**Expected output:**
```
Building for debugging...
Build complete! (X,XX sec)
```

**Watch out for:** If you don't see `Build complete!`, don't proceed to
anything else — fix the build error first. If you see a warning
(`warning:`), take note but you can continue; Command 2 will catch
warnings.

---

### Command 2 — Strict Concurrency + Zero Warnings (THE MOST IMPORTANT BUILD CHECK)
```bash
swift build -Xswiftc -strict-concurrency=complete -Xswiftc -warnings-as-errors
```
**What it tests:** Whether the project still compiles when switched to
Swift 6 language mode, AND that there isn't a single compiler warning left
in the project (`-warnings-as-errors` turns every warning into an error).
Since this project makes heavy use of `actor`, `@unchecked Sendable`, and
raw pointers (`UnsafeMutablePointer`), this check is especially critical —
data race risks are caught here.

**Expected output:**
```
Building for debugging...
Build complete! (X,XX sec)
```
**Not even a single extra line should appear** — if you see any line
containing `warning:`, this command has FAILED (`-warnings-as-errors`
should have already converted it into an error and halted the build, but
check the output anyway just to be sure).

**Watch out for:** This project has previously stumbled on this command
twice (a `sending` data-race error and a `SendableClosureCaptures`
warning — both in the parallel search code in `VectorDatabase.searchBatch`). If
this command produces an error, copy the exact file/line and the full
message it references, verbatim.

---

### Command 3 — Release build (needed for performance measurements)
```bash
swift build -c release -Xswiftc -Ounchecked
```
**What it tests:** Whether the production/performance build compiles.
**Why `-Ounchecked` is passed manually:** This package deliberately does
not embed `-Ounchecked` into `Package.swift` (if it did, the package
couldn't be added as an SPM dependency to another project — SwiftPM
rejects packages containing "unsafe build flags" as versioned
dependencies). So whenever you want a performance measurement, YOU need to
add this flag yourself.

**Expected output:**
```
Building for production...
Build complete! (X,XX sec)
```

**Watch out for:** It also compiles without `-Ounchecked` (`swift build -c
release` on its own), but the runtime performance is then **not
comparable** to the numbers in `BENCHMARKS.md` — bounds/overflow checks
remain active. If you're going to do a comparison, make sure to add
`-Ounchecked`.

---

## 1.2 Test Suite

### Command 4 — Standard (fast) test suite
```bash
swift test
```
**What it tests:** All ~151 tests except two (the 50k recall test and the
energy-profile test — these are skipped by default).

**Expected output (last line):**
```
Executed 153 tests, with 2 tests skipped and 0 failures (0 unexpected) in XXX.XX (XXX.XX) seconds
```

**Watch out for:**
- `with 2 tests skipped` is **expected behavior**, not an error — the
  skipped tests are `testRecallAt50k` (slow, enabled with a separate flag
  — Command 5) and `testEnergyProfileHNSWIOSDevice` (only meaningful in
  Instruments on a physical device, always skipped in automated testing).
- If you see any number other than `0 failures`, find the lines starting
  with `error:` in the output and which test class failed — usually the
  line `Test Case '-[VectorDatabaseTests.X testY]' failed` shows at the top
  which test blew up.
- Total duration varies by machine but shouldn't exceed single-digit
  minutes (typically 5–11 minutes for this project).

---

### Command 5 — FULL suite including slow tests
```bash
RUN_SLOW_TESTS=1 swift test
```
**What it tests:** Everything Command 4 covers, plus `testRecallAt50k`
(an actual HNSW recall verification with 50,000 vectors).

**Expected output:**
```
Executed 153 tests, with 1 test skipped and 0 failures (0 unexpected) in XXX.XX seconds
```
(Now only the energy-profile test is skipped, `1 test skipped`.)

**Watch out for:** This command runs in **debug mode** (Onone), without
`-Ounchecked` — so the 50k insert can take several minutes rather than
seconds in the real world (this is normal; bounds-checking overhead is
high in debug mode). Total run time can reach 10–15 minutes, be patient.

---

### Command 6 — Running a single test class
```bash
swift test --filter HNSWCorrectnessTests
swift test --filter PersistenceTests
swift test --filter WALDurabilityTests
```
**What it tests:** Only the tests in the class you specify. Use this to
quickly verify the relevant area after making a change (without waiting
for the whole suite).

**Expected output:** The `Executed N tests, ... 0 failures` line for that
class.

**Watch out for:** If you misspell the class name (`--filter` is a regex),
it silently returns "Executed 0 tests" — this doesn't look like an error
but actually means nothing ran at all. If you see `Executed 0 tests` in
the output, check the filter name.

---

### Command 7 — Running a single test
```bash
swift test --filter HNSWCorrectnessTests/testRecallAt10k
```
**What it tests:** Only the specified single test. This is the fastest
feedback loop when debugging one test.

**Expected output:** A single `Test Case ... passed` line.

---

### Command 8 — Saving verbose output to a file
```bash
swift test --filter HNSWCorrectnessTests 2>&1 | tee test-output.log
```
**What it tests:** The same thing, but it both displays on screen and
saves to a `test-output.log` file the `print()` output from within the
tests (things like recall values, timing information).

**Watch out for:** Do NOT commit the `test-output.log` file to the repo —
it's a run artifact, not source code. It's recommended you add `*.log` to
`.gitignore` (the repo currently has an untracked `benchmark-after-fix.log`
that made it into the zip, falling into the same category).

---

### Command 9 — Disabling parallel execution (to isolate flaky tests)
```bash
swift test --no-parallel
```
**What it tests:** Runs the tests serially (not in parallel). If you
normally see a test that occasionally fails under parallel execution but
always passes on its own (suspected flaky test), use this — if it always
passes in serial mode, the problem isn't in the test itself but in the
resource contention created by parallel execution (e.g. two tests using
the same temp file path).

---

## 1.3 Sanitizers (The Most Critical Verification Layer in This Project)

This project uses `pthread_rwlock`, `mmap`, raw pointers, and concurrent
access via an `actor` — sanitizers here aren't "nice to have," they're
**mandatory**.

### Command 10 — Thread Sanitizer (targeted, fast)
```bash
TSAN_OPTIONS="halt_on_error=1" swift test --sanitize=thread \
  -Xswiftc -strict-concurrency=complete \
  --filter "VectorDatabaseTests\.(ConcurrencyTests|EndToEndTests|FileLockTests|SearchBatchTests|PersistenceTests)/"
```
**What it tests:** Data races — situations where two threads access the
same memory concurrently without synchronization. Specifically for this
project: concurrent inserts during `save()`, whether `ReadWriteLock`
really provides mutual exclusion, whether `searchBatch`'s parallel workers
step on each other.

**Expected output:**
```
Executed 24 tests, with 0 failures (0 unexpected) in XXX.XX seconds
```

**Watch out for:**
- The `--filter` regex must be exactly in this form (with the
  `VectorDatabaseTests\.(...)/`prefix) — otherwise, since the module name
  (`VectorDatabaseTests`) already appears in every test name, the filter
  effectively becomes "run everything" (this has happened before in this
  project; it's harmless but multiplies the runtime by ~5x).
- If TSan finds a data race, the output starts with `WARNING:
  ThreadSanitizer: data race` and shows exactly which two threads
  conflicted and at which lines — never ignore this output,
  `halt_on_error=1` will already stop at the first race.
- This command is **much slower** than regular `swift test`
  (instrumentation overhead), it can take 5–6 minutes.

---

### Command 11 — Thread Sanitizer (FULL suite, most comprehensive but slowest)
```bash
TSAN_OPTIONS="halt_on_error=1" swift test --sanitize=thread
```
**What it tests:** ALL 153 tests under TSan. This can catch an unexpected
concurrency issue in another test class that Command 10's narrow filter
might miss.

**Expected output:** `Executed 153 tests, with 2 tests skipped and 0 failures`

**Watch out for:** Can take 20–25 minutes. Run this not on every commit,
but after a significant concurrency change or before a release.

---

### Command 12 — Address Sanitizer (essential for mmap/pointer code)
```bash
swift test --sanitize=address
```
**What it tests:** Memory safety violations — out-of-bounds access
(buffer overflow), use-after-free, errors in `MappedFile`'s
`mmap`/`munmap` calls. TSan catches concurrency errors, while ASan catches
memory errors that **can occur even on a single thread** — they aren't
substitutes for each other.

**Expected output:** `Executed 153 tests, with 2 tests skipped and 0 failures`

**Watch out for:** This project doesn't run ASan in CI (only TSan is set
up) — meaning this command is the **only verification layer** covering a
category CI misses. Be sure to run it before releases. Since the `nodes`
dictionary → dense array change involves manual index arithmetic
(`Int(internalID)`, array growth), this command is **especially important**
after that change.

---

### Command 13 — Undefined Behavior Sanitizer
```bash
swift test -Xswiftc -sanitize=undefined
```
**What it tests:** Undefined behavior — integer overflow, unaligned memory
access, and similar issues. Less critical than the other two, but a free
extra layer of checking.

**Expected output:** `Executed 153 tests, with 2 tests skipped and 0 failures`

---

## 1.4 Coverage (Code Coverage)

### Command 14 — Running tests with coverage + report
```bash
swift test --enable-code-coverage && \
BIN_PATH=$(swift build --show-bin-path) && \
XCTEST_BUNDLE=$(find "$BIN_PATH" -name "*.xctest" -type d | head -n1) && \
[ -n "$XCTEST_BUNDLE" ] && \
BINARY_NAME=$(basename "$XCTEST_BUNDLE" .xctest) && \
xcrun llvm-cov report "$XCTEST_BUNDLE/Contents/MacOS/$BINARY_NAME" \
  -instr-profile .build/debug/codecov/default.profdata \
  -ignore-filename-regex="Tests/|.build/"
```
**What it tests:** What percentage of the source code is exercised by the
tests (line/function/branch coverage ratio).

**Expected output:** One line per `.swift` file, with the overall
percentage on the `TOTAL` line. For this project, overall coverage above
`80%` is expected (153 tests, a small codebase).

**Watch out for:** When you look at files with a low `%`, you'll usually
see: error paths (`catch` branches that never trigger after a `throw`) or
rarely-executed defensive code (things like `guard let ... else {
fatalError(...) }`). These don't necessarily mean "missing tests" — but a
file below `50%` is worth investigating to find out why it's low.

---

## 1.5 Benchmarks and Performance Regression

### Command 15 — Recall regression script (fast, for daily use)
```bash
./scripts/benchmark-regression.swift
```
**What it tests:** Compares the recorded recall values in
`benchmarks-baseline.json` (for the 1k and 10k datasets) against the
recall the current code produces. Fails if recall drops below a
threshold.

**Expected output:**
```
🚀 Starting Benchmark Regression Test...
📊 Baseline Recall for testRecallAt1k: 1.0
📈 Current Recall for testRecallAt1k: 1.0
✅ testRecallAt1k passed. No significant regression.
📊 Baseline Recall for testRecallAt10k: 1.0
📈 Current Recall for testRecallAt10k: 1.0
✅ testRecallAt10k passed. No significant regression.
✅ Benchmark passed. No significant regression across 2 dataset size(s).
```

**Watch out for:** This script ONLY checks recall (accuracy), **not
performance (speed)** — there's no latency/throughput data at all in
`benchmarks-baseline.json`. So even if this command passes, it is NOT
proof that a change hasn't slowed down insert/search speed. Use Command
16 for speed regression and compare the numbers by hand.

---

### Command 16 — Full benchmark suite (the real performance evidence)
```bash
swift build -c release -Xswiftc -Ounchecked
swift run -c release -Xswiftc -Ounchecked VectorDatabaseBenchmarks 2>&1 \
  | tee benchmark-$(date +%Y%m%d-%H%M%S).log
```
**What it tests:** vDSP/Accelerate speedups, a 1M-vector insert, a 50k DoD
(definition-of-done) recall + determinism test, a 500k HNSW-vs-Flat
scalability comparison, and an `efSearch` sweep in a realistic
(high-similarity) NLP embedding scenario.

**Expected output (summary, see `BENCHMARKS.md` for the full example):**
```
>>> Speedup: 7.68x  (naive / vDSP)
>>> Batch speedup: 3.54x  (looped vDSP / sgemv)
Insert 1,000,000 vectors (dim=384): ~300ms
DoD Recall@10 check: ✅ PASS  (actual: ~0.97+)
Determinism (1000-node sample): ✅ byte-identical
HNSW Insert 500k vectors: [LOOK HERE — see the note below]
HNSW Speedup at 500k scale: [X.Xx relative to FlatIndex]
efSearch Sweep: Recall@10: 1.0000 on each line
```

**Watch out for — CRITICAL NOTE SPECIFIC TO THIS PROJECT:**
`HNSWIndex.nodes` was recently converted from a dictionary to a dense
array (`nodeSlots`), with the goal of improving insert throughput at large
scale (500k+). Reference value measured before the change:
```
HNSW Insert 500k vectors: 2553571.6466ms  (195.8 vectors/sec)
```
Compare the **new** `HNSW Insert 500k vectors:` line that comes out when
you run this command against this reference value:
- If it's noticeably lower (e.g. below 1.5–2 million ms) → the
  optimization worked; save the file under a meaningful name like
  `benchmark-after-nodemap-optimization.log`.
- If it's nearly the same (around 2.4–2.6 million ms) → the real
  bottleneck isn't the `nodes` dictionary after all (probably the
  O(efConstruction × M) inner loop of `selectNeighborsHeuristic`); note
  this as the next performance task.
- If either the `Recall@10` or `Determinism` lines changed (they were
  previously all ✅/byte-identical) → this is a REGRESSION, the `nodes`
  change has introduced a correctness bug, investigate immediately.

---

## 1.6 iOS Simulator Build

### Command 17
```bash
xcodebuild build -scheme VectorDatabase -destination 'generic/platform=iOS Simulator'
```
**What it tests:** Whether the package also compiles for the actual
target platform, iOS Simulator, not just macOS. `swift build` only
compiles for the host platform (macOS) — an iOS-specific API
incompatibility (e.g. an API below the `deploymentTarget`) is only caught
by this command.

**Expected output (last line):**
```
** BUILD SUCCEEDED **
```

**Watch out for:** The output is very long (hundreds of lines — separate
build steps for each module, precompiled module cache operations). If you
don't care about the details, use this instead:
```bash
xcodebuild build -scheme VectorDatabase -destination 'generic/platform=iOS Simulator' 2>&1 | tail -5
```
If you see `** BUILD FAILED **`, search upward for the first `error:`
line — it usually appears right below a `SwiftCompile` or
`SwiftEmitModule` step.

---

## 1.7 Cleanup / Troubleshooting

```bash
# Delete build artifacts (guarantees the next build is clean)
swift package clean

# Fully reset .build (more aggressive, use for cache issues)
swift package reset

# Clear Xcode's DerivedData cache (if you're getting weird errors via Xcode)
rm -rf ~/Library/Developer/Xcode/DerivedData/*

# Clean up leftover temporary .vdb/.wal/.lock files from tests
rm -f $TMPDIR*.vdb $TMPDIR*.wal $TMPDIR*.lock $TMPDIR*.tmp
```
**When to use:** If a command fails "inexplicably" (especially with errors
like "file already exists" or "module not found"), try `swift package
clean` first, and if that doesn't help, move on to `reset`.

---

## 1.8 Full Verification — The Complete Sequence to Run Before Release

```bash
set -e
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

swift package clean
swift build -Xswiftc -strict-concurrency=complete -Xswiftc -warnings-as-errors
RUN_SLOW_TESTS=1 swift test --enable-code-coverage
TSAN_OPTIONS="halt_on_error=1" swift test --sanitize=thread \
  --filter "VectorDatabaseTests\.(ConcurrencyTests|EndToEndTests|FileLockTests|SearchBatchTests|PersistenceTests)/"
swift test --sanitize=address
./scripts/benchmark-regression.swift
swift build -c release -Xswiftc -Ounchecked
swift run -c release -Xswiftc -Ounchecked VectorDatabaseBenchmarks | tee benchmark-$(date +%Y%m%d).log
xcodebuild build -scheme VectorDatabase -destination 'generic/platform=iOS Simulator'
echo "✅ All passed"
```
**Watch out for:** Thanks to `set -e`, if any step fails, the script stops
right there — don't skip any step without seeing the "✅ All passed"
message. Total duration can be between **30–50 minutes** depending on your
machine (faster than Command 11 since the full TSan suite isn't included,
but still long); a good command to run during a coffee break.

---

# SECTION 2 — Testing via the Xcode Interface

## 2.1 Opening the Project

In the terminal:
```bash
open Package.swift
```
or double-click `Package.swift` in Finder. Xcode automatically opens
`Package.swift` as a project (a separate `.xcodeproj` isn't needed — this
project already comes with scheme/test-plan files ready under
`.swiftpm/xcode/`).

**Watch out for:** The scheme picker in the top left will show three
options: `swift-vector-database-Package`, `VectorDatabase`, `VectorDatabaseBenchmarks`.
`swift-vector-database-Package` or `VectorDatabase` must be selected to run tests; select
`VectorDatabaseBenchmarks` to run benchmarks.

---

## 2.2 Cmd+U — Running All Tests

**⚠️ VERY IMPORTANT — THERE'S A TRAP SPECIFIC TO THIS PROJECT:**

This project's `.swiftpm/swift-vector-database-Package.xctestplan` file comes with
the following two settings **pre-enabled** by default:

```json
"environmentVariableEntries": [{ "key": "RUN_SLOW_TESTS", "value": "1" }],
"threadSanitizerEnabled": true
```

That means **the moment you press Cmd+U**, even if you haven't selected
anything, Xcode will:
1. Also run `testRecallAt50k` (same as Terminal Command 5),
2. Automatically run with Thread Sanitizer on (same instrumentation as
   Terminal Command 10/11).

This is equivalent to the terminal command `RUN_SLOW_TESTS=1
TSAN_OPTIONS=... swift test --sanitize=thread` — meaning it is **NOT a
quick check**, but the most comprehensive and slowest run. It can take
20–30+ minutes.

**What it tests:** The entire `VectorDatabaseTests` target in the test plan,
under TSan, including the slow tests.

**Expected output:** Every test in the Test Navigator (Cmd+6) on the left
panel gets marked with a green checkmark. "Test Succeeded" message at the
top. Total duration and test count appear in the Report Navigator
(Cmd+9) (`153 tests`).

**If you want a quick check** (during day-to-day development, just to
check "did I break something"), you need to temporarily edit the test
plan:

1. Click the `swift-vector-database-Package.xctestplan` file in the project
   navigator (or `Product > Test Plan > Edit Test Plan...`).
2. In the **Configurations** tab, delete the `RUN_SLOW_TESTS` line or set
   its value to `0`.
3. Turn off **Thread Sanitizer** under **Options**.
4. Cmd+U — this is now equivalent to Terminal Command 4 (`swift test`),
   finishes in a few minutes.
5. **Be sure to turn these back on before release** — these settings were
   deliberately left this way so that "the most comprehensive run is the
   default"; don't make the temporary change permanent.

---

## 2.3 Test Navigator (Cmd+6) — Running a Single Test / Single Class

**What it does:** Pressing Cmd+6 lists all test classes and methods as a
tree in the left panel. Clicking the diamond (◇) icon to the left of any
row runs only that test/class.

**Usage:**
- Click the diamond next to a class name → only the tests in that class
  run (equivalent to Terminal Command 6).
- Click the diamond next to a method → only that test runs (equivalent to
  Terminal Command 7).
- While a test is running, the diamond turns into a spinning wheel, and
  once finished becomes a green checkmark (✓) or a red X (✗).

**Watch out for:** Even a single test run via the Test Navigator **still
uses the active test plan's settings** (the TSan/RUN_SLOW_TESTS warning
above applies here too). Only "which tests run" is filtered, not "how they
run."

---

## 2.4 Running via the Diamond Icons in the Editor

When you open a test file (e.g. `HNSWCorrectnessTests.swift`) in the
editor, a diamond icon appears to the left of every `func testXXX()` line.

**What it does:** Clicking the diamond on that line runs only that test —
without going to the Test Navigator, for quick feedback while writing
code. The diamond next to the class definition (`class
HNSWCorrectnessTests`) runs ALL the tests in that class.

**Watch out for:** If a test passes, the diamond turns green and stays on
the left of the line — next time you can click it to re-run just that
test. If a test fails, it turns into a red X AND an inline red error
message appears on the exact line where the failure occurred (something
like `XCTAssertEqual failed: ("X") is not equal to ("Y")`) — this is the
same information as the terminal output, just shown line-by-line within
the code.

---

## 2.5 Report Navigator (Cmd+9) — Reading Results and Coverage

**What it does:** The history of every test/build run accumulates here.
Clicking on a run shows:
- **Top left:** a general summary (number of tests, number of failures,
  total duration).
- If there's a failed test, clicking it jumps straight to the code where
  the failure occurred.
- **Coverage tab** (if coverage was enabled before `Cmd+U` — see 2.6):
  shows line/function coverage percentage for each file; clicking a file
  shows which lines never ran (highlighted in red).

**Watch out for:** When TSan finds a data race, the result is NOT `Test
Succeeded`, but Xcode sometimes shows this as a separate "Issue" that may
not look like a classic red test failure. Even if the report summary says
"0 failures," it's safer to check by searching for the text `WARNING:
ThreadSanitizer` in the console output (Cmd+9 → relevant run → console
icon in the top right).

---

## 2.6 Enabling Code Coverage in Xcode

**Steps:**
1. `Product > Scheme > Edit Scheme...` (or Cmd+<).
2. Select the **Test** action in the left panel.
3. In the **Options** tab, check the **Code Coverage** box, and select the
   `VectorDatabase` target under "Gather coverage for."
4. Run the tests with Cmd+U.
5. Cmd+9 (Report Navigator) → latest run → **Coverage** tab in the top
   middle bar.

**Expected output:** One line and one percentage per `.swift` file. Shows
the same data as Terminal Command 14, in visualized form.

---

## 2.7 Running the Benchmark Target in Xcode

**⚠️ SECOND IMPORTANT TRAP — ALSO SPECIFIC TO THIS PROJECT:**

The `VectorDatabaseBenchmarks.xcscheme` file's Run (▶️) action comes **set to
the `Debug` configuration by default** — not `Release`.

That means when you select `VectorDatabaseBenchmarks` from the scheme picker and
**press the ▶️ (Run) button**, the benchmarks run in Debug mode (Onone, no
`-Ounchecked`). This is equivalent to the terminal command `swift run
VectorDatabaseBenchmarks` (WITHOUT the release flag) — the numbers are **not
comparable to the figures in BENCHMARKS.md and TEST_GUIDE §1.5**, they
come out much slower (e.g. a 50k insert takes minutes in debug mode,
~38 seconds in release).

**For a correct performance measurement:**
1. `Product > Scheme > Edit Scheme...` (Cmd+<).
2. Select the **Run** action in the left panel.
3. In the **Info** tab, switch **Build Configuration** from `Debug` to
   `Release`.
4. Close, run with ▶️.

**Note:** This still doesn't add `-Ounchecked` (that flag isn't the
default for Xcode's Release configuration, and it's deliberately absent
from Package.swift too — see the explanation of §1.1 Command 3). If you
want to add `-Ounchecked` via Xcode: in the **Build Settings** tab, you
can add `-Ounchecked` to `Other Swift Flags`, only on the `Release` line.
But the most reliable, most easily repeatable way is still **Terminal
Command 16** — running benchmarks via Xcode is fine for a daily "quick
look," but not for an official/comparison measurement.

**What it tests / Expected output:** Identical to Terminal Command 16
(the same `main.swift` runs) — the output is printed to the console in
Xcode's bottom panel (View > Debug Area > Activate Console, or
Cmd+Shift+Y).

---

## 2.8 Leak / Performance Profiling with Instruments

The "1M inserts for Instruments Leak test" section in the benchmark file,
as the name suggests, gets its real verification inside **Instruments**,
not just by looking at console output.

**Steps:**
1. Select `VectorDatabaseBenchmarks` from the scheme picker (it's recommended
   you switch it to Release as in §2.7, for a realistic profile).
2. `Product > Profile...` (Cmd+I).
3. In the Instruments template picker that opens, select the **Leaks**
   template (for memory leaks) or **Time Profiler** (to see which
   function eats the most CPU time — this is the MOST ACCURATE way to see
   whether the `nodes` optimization actually solved the bottleneck, based
   on real measurement rather than guesswork like in
   `benchmark-after-fix.log`).
4. Press the red ● (Record) button — the benchmark starts running.
5. **In the Leaks template:** If a red number appears in the top right
   (leak count) — it should be 0. It should still be 0 after the `1M
   inserts` section finishes.
6. **In the Time Profiler template:** Check the "Invert Call Tree" and
   "Hide System Libraries" boxes in the bottom left "Call Tree" panel —
   the functions consuming the most time are listed at the top. To see
   the effect of the `nodes` optimization, look at the percentage share
   of the `HNSWIndex.insert`, `searchLayer`, and
   `selectNeighborsHeuristic` functions.

**Watch out for:** Instruments adds some instrumentation overhead even in
the Release configuration (particularly depending on Time Profiler's
sampling rate) — the absolute times you see here may not be exactly
identical to `swift run -c release`, but the RELATIVE distribution across
functions (what percentage of total time each one eats) is reliable, and
that's the information you're actually after anyway.

---

## 2.9 Building for iOS Simulator (via Xcode)

1. Select `VectorDatabase` from the scheme picker.
2. Select an iOS Simulator (e.g. "iPhone 17") from the device picker right
   next to it (to the right of the scheme picker).
3. Cmd+B (build only, don't run — this is a library target, it can't be
   run directly).

**Expected output:** Green checkmark + "Build Succeeded" in the top left.
Verifies the same result as Terminal Command 17, just from the graphical
interface.

**Watch out for:** If you press Cmd+B without selecting a simulator (while
host/My Mac is selected), it only compiles for macOS — it will NOT catch
an iOS-specific issue. Make sure the device picker really says "iPhone ...
Simulator."