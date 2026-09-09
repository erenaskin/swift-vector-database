import SwiftUI
import VectorDB

struct EnergyProfilerView: View {
    @State private var status = "Ready to Profile"
    @State private var isRunning = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("HNSW Energy Profiler")
                .font(.headline)
            
            Text(status)
                .foregroundColor(isRunning ? .blue : .primary)
                .multilineTextAlignment(.center)
                .padding()
            
            Button(action: {
                runWorkload()
            }) {
                Text(isRunning ? "Running..." : "Start 15s Workload")
                    .padding()
                    .background(isRunning ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .disabled(isRunning)
        }
        .padding()
    }
    
    func runWorkload() {
        isRunning = true
        status = "Workload is running...\nCheck Instruments Energy Log!"
        
        Task {
            let dim = 128
            // Initialize VectorDB
            let db = try! VectorDB(dimension: dim, metric: .cosine)
            
            let start = Date()
            var insertCount = 0
            var searchCount = 0
            
            // Sustain CPU workload for exactly 15 seconds
            while Date().timeIntervalSince(start) < 15.0 {
                // Insert batch
                for i in 0..<100 {
                    let vec = (0..<dim).map { _ in Float.random(in: -1...1) }
                    try! await db.insert(id: "doc_\(insertCount + i)", vector: vec)
                }
                insertCount += 100
                
                // Search batch
                for _ in 0..<50 {
                    let query = (0..<dim).map { _ in Float.random(in: -1...1) }
                    _ = try! await db.search(query: query, k: 5)
                    searchCount += 1
                }
            }
            
            status = "Completed!\nInserts: \(insertCount)\nSearches: \(searchCount)"
            isRunning = false
        }
    }
}
