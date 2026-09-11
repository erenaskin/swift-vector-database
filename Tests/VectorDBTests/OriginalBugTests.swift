import XCTest
@testable import VectorDB

final class OriginalBugTests: XCTestCase {

    var dbURL: URL!

    override func setUp() {
        super.setUp()
        let tempDir = FileManager.default.temporaryDirectory
        dbURL = tempDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("vdb")
    }

    override func tearDown() {
        if FileManager.default.fileExists(atPath: dbURL.path) {
            try? FileManager.default.removeItem(at: dbURL)
        }
        let walURL = dbURL.appendingPathExtension("wal")
        if FileManager.default.fileExists(atPath: walURL.path) {
            try? FileManager.default.removeItem(at: walURL)
        }
        super.tearDown()
    }

    // B4 Kapsam Netleştirme: Runtime delete() sorunsuz, sadece WAL replay bug'lı.
    func testDeleteWALReplayBugBlastRadius() async throws {
        let db1 = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        
        // 1. Kayıt ekle ve SAVE ile kalıcı yap (IDMap disk'e yazılsın)
        try await db1.insert(id: "bug1", vector: [1.0, 1.0])
        try await db1.save()
        
        // 2. Canlı (runtime) delete yap
        try await db1.delete(id: "bug1")
        
        // Canlı delete başarılı mı? Evet, idMap.remove() çalıştı.
        let liveGet = await db1.get(id: "bug1")
        XCTAssertNil(liveGet, "Runtime delete basarisiz!") // Bu geçer (çünkü canlıda sorun yok)
        
        // Şimdi crash simülasyonu yapıyoruz (save yapmadan db2 aç)
        // db2 açıldığında, snapshot'ta (save anında) "bug1" idMap içindeydi.
        // WAL replay sırasında ".delete" okunacak ama PersistenceManager.load 
        // idMap'ten SILMEYECEK.
        let db2 = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        
        // 3. Blast Radius testi: Aynı externalID'yi tekrar eklemeye çalışırsak ne olur?
        do {
            // FIX UYGULANDI: Artık WAL replay `.delete` işlemi idMap'ten de kaydı
            // sildiği için insert başarılı olmalı. duplicateID Fırlatmamalı!
            try await db2.insert(id: "bug1", vector: [2.0, 2.0])
            
            // Başarılı olduysa testi geç. Get ile kontrol edelim.
            let successGet = await db2.get(id: "bug1")
            XCTAssertEqual(successGet?.vector, [2.0, 2.0])
        } catch {
            XCTFail("Insert başarisiz oldu, idMap tam temizlenmemis olabilir: \(error)")
        }
    }
    
    // B2 Açık Soru: Insert-time metadata (ve aslinda externalID) crash sonrasi kayboluyor mu?
    func testInsertMetadataLostOnCrash() async throws {
        let db1 = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        
        // 1. Snapshot/save YOK. Sadece insert yapiyoruz.
        try await db1.insert(id: "lost1", vector: [1.0, 1.0], metadata: ["key": "val"])
        
        // 2. Crash simülasyonu (save() cagrilmadan db2'yi ac).
        let db2 = try VectorDB(dimension: 2, metric: .dotProduct, path: dbURL)
        
        // 3. db2'de kaydi arayalim.
        let result = await db2.get(id: "lost1")
        
        // SONUC: Sadece metadata değil, kaydin TAMI (externalID dahil) idMap'te yok!
        // WAL sadece internalID ve vector tutuyor, idMap ise sadece save() ile JSON olarak tutuluyor.
        // Bu yüzden result tamamen nil gelecektir.
        XCTAssertNil(result, "Kayıt idMap'ten tamamen kaybolmustur, dolayisiyla result nil'dir.")
        print("B2 KANITLANDI: Insert sirasinda verilen metadata (ve String ID) save() yapilmadan crash olursa TAMAMEN kaybolur.")
    }
}
