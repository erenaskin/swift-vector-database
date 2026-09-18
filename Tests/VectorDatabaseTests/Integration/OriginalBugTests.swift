import XCTest

@testable import VectorDatabase

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
        let db1 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        // 1. Kayıt ekle ve SAVE ile kalıcı yap (IDMap disk'e yazılsın)
        try await db1.insert(id: "bug1", vector: [1.0, 1.0])
        try await db1.save()

        // 2. Canlı (runtime) delete yap
        try await db1.delete(id: "bug1")

        // Canlı delete başarılı mı? Evet, idMap.remove() çalıştı.
        let liveGet = await db1.get(id: "bug1")
        XCTAssertNil(liveGet, "Runtime delete basarisiz!")  // Bu geçer (çünkü canlıda sorun yok)

        // Şimdi crash simülasyonu yapıyoruz (save yapmadan db2 aç)
        // db2 açıldığında, snapshot'ta (save anında) "bug1" idMap içindeydi.
        // WAL replay sırasında ".delete" okunacak ama PersistenceManager.load
        // idMap'ten SILMEYECEK.
        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

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

    func testInsertSurvivesCrashWithoutSave() async throws {
        let db1 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        // 1. Snapshot/save YOK. Sadece insert yapiyoruz. (Metadatasız)
        try await db1.insert(id: "survive1", vector: [1.0, 1.0])

        // 2. Crash simülasyonu (save() cagrilmadan db2'yi ac).
        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        // 3. db2'de kaydi arayalim.
        let result = await db2.get(id: "survive1")

        XCTAssertNotNil(
            result,
            "Kayıt (externalID ile birlikte) WAL replay sayesinde başarıyla kurtarılmalıdır.")
        XCTAssertEqual(result?.vector, [1.0, 1.0])
    }

    func testInsertWithMetadataSurvivesCrashWithoutSave() async throws {
        let db1 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        // 1. Snapshot/save YOK. Sadece metadata'lı insert yapiyoruz. (.insertWithMetadata opcode)
        try await db1.insert(id: "lost1", vector: [1.0, 1.0], metadata: ["key": "val"])

        // 2. Crash simülasyonu (save() cagrilmadan db2'yi ac).
        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)

        // 3. db2'de kaydi arayalim.
        let result = await db2.get(id: "lost1")

        XCTAssertNotNil(result, "Kayıt WAL replay sayesinde başarıyla kurtarılmalıdır.")
        XCTAssertEqual(
            result?.metadata?["key"], "val", "Metadata WAL üzerinden başarıyla yüklenmelidir.")
    }

    func testTruncateCrashWindowDoesNotDoubleReplay() async throws {
        let db1 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        try await db1.insert(id: "vec1", vector: [1.0, 1.0])

        let mirror = Mirror(reflecting: db1)
        guard
            let pm = mirror.children.first(where: { $0.label == "persistenceManager" })?.value
                as? PersistenceManager
        else {
            XCTFail("PersistenceManager bulunamadı")
            return
        }

        // Atomic rename sonrası, ama truncate öncesi crash hook'u
        pm.onAfterRename = {
            // Test ortamında burada execution'ı kesemediğimiz için, WAL dosyası bilerek truncate edilmeyecekmiş
            // gibi simüle edeceğiz veya save()'in bitmesine izin verip sonra `db2` ile test edeceğiz.
            // Ama db1.save() başarılı olursa `wal.truncate()` çalışır.
            // Bu yüzden testi özel şekilde yapalım:
        }

        // Burada gerçek bir onAfterRename exception fırlatarak save'i yarıda keseceğiz
        pm.onAfterRename = {
            // We just throw a fatal error or a generic error to break out?
            // Actually, we can't throw from onAfterRename because it doesn't throw.
            // Let's modify PersistenceManager to allow throwing, or we just rely on a testing trick:
            // Actually, if we just copy the WAL file contents before it truncates!
        }

        // For testing, let's copy the WAL file in the hook to simulate crash
        let crashedWALURL = dbURL.appendingPathExtension("crashed_wal")
        let localDBURL = dbURL!
        pm.onAfterRename = {
            try? FileManager.default.copyItem(
                at: localDBURL.appendingPathExtension("wal"), to: crashedWALURL)
        }

        try await db1.save()

        // Simülasyon: Crash oldu. .vdb yeni versiyonda. Ama WAL dosyası truncate EDİLMEDİ (crashedWALURL'u geri kopyalıyoruz).
        try FileManager.default.removeItem(at: self.dbURL.appendingPathExtension("wal"))
        try FileManager.default.moveItem(
            at: crashedWALURL, to: self.dbURL.appendingPathExtension("wal"))

        // db2'yi aç. Hem .vdb'den okuyacak, hem de WAL'dan eski `.insert` kaydını bulacak.
        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        let stats = await db2.stats()

        // Double-replay koruması (idMap.contains) sayesinde liveCount = 1 kalmalı, şişmemeli!
        XCTAssertEqual(stats.liveCount, 1, "Double replay engellenmelidir, liveCount şişmemelidir.")
    }

    func testHNSWParametersSurviveReload() async throws {
        let params = HNSWParameters(M: 12, efConstruction: 300, efSearch: 200, seed: 0xABCD)
        let db1 = try VectorDatabase(
            dimension: 2, metric: .dotProduct, parameters: params, path: dbURL)
        try await db1.insert(id: "v1", vector: [1, 1])
        try await db1.save()

        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        let mirror = Mirror(reflecting: db2)
        guard let engine = mirror.children.first(where: { $0.label == "engine" })?.value as? Engine
        else {
            XCTFail()
            return
        }

        let index = engine.exportHNSWIndex()

        XCTAssertEqual(index.params.efConstruction, 300)
        XCTAssertEqual(index.params.efSearch, 200)
        XCTAssertEqual(index.params.seed, 0xABCD)
    }

    func testLegacyWALWithoutHeaderThrowsOnInsertReplay() async throws {
        let walURL = dbURL.appendingPathExtension("wal")

        // Elle v0 (header'sız) bir WAL dosyası oluşturalım:
        // Record: opcode(0) + id(0) + ts(0) + vectorData([1.0, 1.0]) (dimension=2)
        var data = Data()
        var opcode: UInt8 = 0  // .insert
        data.append(&opcode, count: 1)
        var id: Int32 = 0
        data.append(withUnsafeBytes(of: &id) { Data($0) })
        var ts: UInt64 = 0
        data.append(withUnsafeBytes(of: &ts) { Data($0) })
        let vec: [Float] = [1.0, 1.0]
        vec.withUnsafeBufferPointer { buf in
            data.append(buf)
        }

        try data.write(to: walURL)

        // VDB bu legacy WAL'ı okurken .insert opcode'u görünce legacyWALFormatNotSupported hatası fırlatmalı
        do {
            _ = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
            XCTFail(
                "Header'ı olmayan legacy WAL'daki insert record'u sessizce geçilmemeli, hata fırlatmalıdır."
            )
        } catch VectorDatabaseError.legacyWALFormatNotSupported(_) {
            // Başarılı, beklenen hata
        } catch {
            XCTFail("Beklenmeyen hata: \(error)")
        }
    }
}
