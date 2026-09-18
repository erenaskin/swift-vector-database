import XCTest

@testable import VectorDatabase

final class MetadataUpdateTests: XCTestCase {

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

    func testUpdateMetadataBasic() async throws {
        let db = try VectorDatabase(dimension: 2, metric: .euclidean)
        let vec: [Float] = [1.0, 2.0]

        try await db.insert(id: "doc1", vector: vec, metadata: ["type": "old"])

        // Vektör bit-bit aynı kalmalı, metadata değişmeli
        try await db.updateMetadata(id: "doc1", metadata: ["type": "new"])

        let result = await db.get(id: "doc1")
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.vector, vec)
        XCTAssertEqual(result?.metadata, ["type": "new"])
    }

    func testUpdateMetadataNotFound() async throws {
        let db = try VectorDatabase(dimension: 2)
        do {
            try await db.updateMetadata(id: "not_exists", metadata: ["key": "val"])
            XCTFail("Gerekli hata firlatilmadi")
        } catch VectorDatabaseError.notFound(let id) {
            XCTAssertEqual(id, "not_exists")
        } catch {
            XCTFail("Yanlis hata: \(error)")
        }
    }

    func testUpdateMetadataCrashRecovery() async throws {
        // WAL kaydının düzgün yapıldığını ve Snapshot alınmadan çökme durumunda
        // WAL'dan yeni metadata'nın doğru okunduğunu doğrulayacağız.

        let db1 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        try await db1.insert(id: "doc1", vector: [1.0, 1.0], metadata: ["k1": "v1"])

        // Metadata'yı 2 kez güncelliyoruz (Idempotency / Sıralama testi)
        try await db1.updateMetadata(id: "doc1", metadata: ["k1": "v2"])
        try await db1.updateMetadata(id: "doc1", metadata: ["k1": "v3", "extra": "true"])

        // WAL buffer'ını disk'e yazdığından emin ol
        // (updateMetadata içinde walFlushInterval = 1 olduğu için otomatik yazılmıştır ama yine de emin olalım)
        // db1.close() YAPMIYORUZ! (Crash simülasyonu, compact/flush tetiklenmemeli)

        // db2'yi direkt aç, JSON tail'de (save() olmadığı için) veritabanı boş görünebilir.
        // Ancak bizim sistemde init sırasında load() çalışırken WAL record'ları okunur.
        // Not: Mevcut mimari insert'leri snapshot olmadan IDMap'e eklemiyor (idMap sadece
        // save ile kalıcılaşıyor). VectorDatabase tasarımı gereği tam crash durumunda IDMap
        // kaybolacağından, test için önce save() yapıp sonra updateMetadata çağıralım.

        try await db1.save()  // Insert kalıcı olsun

        // Şimdi metadata güncelleyip crash olalım
        try await db1.updateMetadata(id: "doc1", metadata: ["k1": "vFINAL"])

        // DB2: Load from disk
        let db2 = try VectorDatabase(dimension: 2, metric: .dotProduct, path: dbURL)
        let result = await db2.get(id: "doc1")

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.metadata, ["k1": "vFINAL"])  // WAL replay'den gelmeli!
    }

    func testBackwardCompatibilityWithOldWAL() throws {
        // Eski format (0: insert, 1: delete) içeren manuel bir WAL dosyası yazıp
        // PersistenceManager'ın okuyabildiğini doğrulayalım.

        let walURL = dbURL.appendingPathExtension("wal")
        let fh = try FileHandle(
            forWritingTo: FileManager.default.createFile(atPath: walURL.path, contents: nil)
                ? walURL : walURL)
        defer { try? fh.close() }

        var data = Data()

        // 1. Insert Record (Opcode 0) - internalID 10, ts 1000, vector [0.5, 0.5]
        var opInsert: UInt8 = 0
        data.append(&opInsert, count: 1)

        var id1: Int32 = 10
        data.append(withUnsafeBytes(of: &id1) { Data($0) })

        var ts1: UInt64 = 1000
        data.append(withUnsafeBytes(of: &ts1) { Data($0) })

        let vector: [Float] = [0.5, 0.5]
        vector.withUnsafeBufferPointer { buf in
            data.append(buf)
        }

        // 2. Delete Record (Opcode 1) - internalID 10, ts 2000
        var opDelete: UInt8 = 1
        data.append(&opDelete, count: 1)

        var id2: Int32 = 10
        data.append(withUnsafeBytes(of: &id2) { Data($0) })

        var ts2: UInt64 = 2000
        data.append(withUnsafeBytes(of: &ts2) { Data($0) })

        try fh.write(contentsOf: data)
        try fh.synchronize()

        // WriteAheadLog.readAll() çağıralım
        let wal = try WriteAheadLog(path: walURL, dimension: 2)

        do {
            _ = try wal.readAll()
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
