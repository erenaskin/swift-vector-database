#!/usr/bin/env swift

import Foundation

// MARK: - Models

struct BaselineData: Codable {
    struct TestData: Codable {
        let datasetSize: Int
        let recall: Double
    }
    
    // A simple dictionary mapping test name to its data
    // In our case we have "testRecallAt1k" and "testRecallAt10k"
    // We can just decode it into a dictionary
}

/// One baseline entry this script checks against a live `swift test` run.
struct RecallCheck {
    /// Key into benchmarks-baseline.json (also used as the XCTest method name
    /// via `HNSWCorrectnessTests.<testName>`).
    let testName: String
    /// The `datasetSize` argument `runRecallTest` was called with, so we can
    /// match the exact "Recall@10 for dataset size <N>:" line it prints.
    let datasetSize: Int
}

// MARK: - Execution

func runCommand(_ command: String) -> String {
    let task = Process()
    let pipe = Pipe()
    
    task.standardOutput = pipe
    task.standardError = pipe
    task.arguments = ["-c", command]
    task.executableURL = URL(fileURLWithPath: "/bin/zsh")
    
    do {
        try task.run()
        task.waitUntilExit()
    } catch {
        print("Error running command: \(error)")
        exit(1)
    }
    
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return String(data: data, encoding: .utf8) ?? ""
}

/// Runs one baseline's corresponding XCTest, parses its printed recall, and
/// compares it against the stored baseline. Returns `false` (without exiting)
/// on a regression so `main()` can run every check before deciding whether to
/// fail the whole script — one failing dataset size shouldn't hide a second
/// one.
func checkRecall(_ check: RecallCheck, baseline: [String: BaselineData.TestData], developerDirPrefix: String) -> Bool {
    guard let baselineEntry = baseline[check.testName] else {
        print("❌ Could not find \(check.testName) in baseline")
        return false
    }

    print("📊 Baseline Recall for \(check.testName): \(baselineEntry.recall)")

    print("⏳ Running HNSWCorrectnessTests.\(check.testName)...")
    let output = runCommand("\(developerDirPrefix)swift test --filter HNSWCorrectnessTests.\(check.testName)")

    // We are looking for a line like "Recall@10 for dataset size 1000: 1.0"
    let marker = "Recall@10 for dataset size \(check.datasetSize):"
    let lines = output.components(separatedBy: .newlines)
    var newRecall: Double?

    for line in lines {
        if line.contains(marker) {
            let parts = line.components(separatedBy: ":")
            if parts.count >= 2 {
                let valStr = parts.last!.trimmingCharacters(in: .whitespaces)
                newRecall = Double(valStr)
            }
        }
    }

    guard let currentRecall = newRecall else {
        print("❌ Could not parse recall from test output for \(check.testName). Output was:")
        print(output)
        return false
    }

    print("📈 Current Recall for \(check.testName): \(currentRecall)")

    // Compare (5% threshold)
    let threshold = 0.05
    let drop = baselineEntry.recall - currentRecall

    if drop > threshold {
        print("❌ REGRESSION DETECTED in \(check.testName)!")
        print("Recall dropped by \(String(format: "%.2f%%", drop * 100)). Threshold is 5%.")
        return false
    }

    print("✅ \(check.testName) passed. No significant regression.")
    return true
}

func main() {
    print("🚀 Starting Benchmark Regression Test...")
    
    // 1. Load Baseline
    let baselineURL = URL(fileURLWithPath: "benchmarks-baseline.json")
    guard let baselineData = try? Data(contentsOf: baselineURL),
          let baseline = try? JSONDecoder().decode([String: BaselineData.TestData].self, from: baselineData) else {
        print("❌ Could not load benchmarks-baseline.json")
        exit(1)
    }

    let env = ProcessInfo.processInfo.environment
    var developerDirPrefix = ""
    if let devDir = env["DEVELOPER_DIR"] {
        developerDirPrefix = "DEVELOPER_DIR=\(devDir) "
    }

    // Every dataset size defined in benchmarks-baseline.json is checked here.
    // (Previously only testRecallAt1k was checked even though the baseline
    // file also defines testRecallAt10k — that entry was loaded into memory
    // but never read again, so a 10k recall regression could never actually
    // be caught by CI.)
    let checks = [
        RecallCheck(testName: "testRecallAt1k", datasetSize: 1_000),
        RecallCheck(testName: "testRecallAt10k", datasetSize: 10_000),
    ]

    var allPassed = true
    for check in checks {
        let passed = checkRecall(check, baseline: baseline, developerDirPrefix: developerDirPrefix)
        allPassed = allPassed && passed
    }

    if allPassed {
        print("✅ Benchmark passed. No significant regression across \(checks.count) dataset size(s).")
        exit(0)
    } else {
        exit(1)
    }
}

main()
