import XCTest
@testable import VectorDB

final class ReadAPITests: XCTestCase {

    func testEmptyDatabase() async throws {
        let db = try VectorDB(dimension: 2, metric: .cosine)
        
        // Boş DB'de listIDs
        let list1 = await db.listIDs(offset: 0, limit: 10)
        XCTAssertTrue(list1.isEmpty)
        
        // Boş DB'de get(id:)
        let result = await db.get(id: "nonexistent")
        XCTAssertNil(result)
    }
    
    func testGetVectorExactMatch() async throws {
        let db = try VectorDB(dimension: 3, metric: .euclidean)
        
        let originalVector: [Float] = [1.5, 2.5, -3.0]
        let originalMetadata = ["key": "value"]
        try await db.insert(id: "vec1", vector: originalVector, metadata: originalMetadata)
        
        guard let result = await db.get(id: "vec1") else {
            XCTFail("Vector not found")
            return
        }
        
        // Orijinal vektörle bit-bit aynı olduğunu doğrula
        XCTAssertEqual(result.vector, originalVector)
        XCTAssertEqual(result.metadata, originalMetadata)
    }
    
    func testPaginationAndDeletion() async throws {
        // HNSW eşiğini aşmamak için threshold altı (FlatIndex) ve üstü (HNSWIndex)
        // iki durum için de çalışabilmeli, ama limitler genel.
        let db = try VectorDB(dimension: 2, metric: .dotProduct)
        
        let ids = (0..<10).map { "vec\($0)" }
        for (i, id) in ids.enumerated() {
            try await db.insert(id: id, vector: [Float(i), Float(i)])
        }
        
        // Silmeden önce listIDs
        let page1 = await db.listIDs(offset: 0, limit: 5)
        XCTAssertEqual(page1, ["vec0", "vec1", "vec2", "vec3", "vec4"]) // Insertion order
        
        let page2 = await db.listIDs(offset: 5, limit: 10) // Limit > kalan
        XCTAssertEqual(page2, ["vec5", "vec6", "vec7", "vec8", "vec9"])
        
        // Birkaçını sil (örneğin vec2 ve vec7)
        try await db.delete(id: "vec2")
        try await db.delete(id: "vec7")
        
        // Silinenler için get(id:) nil dönmeli
        let getDeleted1 = await db.get(id: "vec2")
        XCTAssertNil(getDeleted1)
        
        let getDeleted2 = await db.get(id: "vec7")
        XCTAssertNil(getDeleted2)
        
        // Silinenler listIDs'de görünmemeli
        let allRemaining = await db.listIDs(offset: 0, limit: 100)
        XCTAssertEqual(allRemaining, ["vec0", "vec1", "vec3", "vec4", "vec5", "vec6", "vec8", "vec9"])
        
        // Offset / Limit sınır durumları
        let outOfBounds = await db.listIDs(offset: 10, limit: 5) // offset > count (count = 8)
        XCTAssertTrue(outOfBounds.isEmpty)
        
        // Limit = 0 veya negatif olursa listExternalIDs boş döner
        let zeroLimit = await db.listIDs(offset: 0, limit: 0)
        XCTAssertTrue(zeroLimit.isEmpty)
    }

    func testHNSWPaginationAndDeletion() async throws {
        // HNSW'ye geçmesi için 2000'den fazla ekleme yapalım veya threshold'u küçük veren
        // bir constructor ile deneyebiliriz ama VectorDB threshold'u parametre almıyor.
        // O yüzden FlatIndex ve HNSWIndex'i mock'lamadan HNSW'yi tetiklemek için testte
        // dbEngine'in 2000 threshold'u geçtiği durumu simüle edebiliriz, ama çok yavaş olabilir.
        // get(id:)'nin HNSW'deki tombstoned davranışını doğrulamak için manuel HNSWIndex testi:
        
        var hnsw = HNSWIndex(dimension: 2, metric: .dotProduct)
        try hnsw.insert(internalID: 0, vector: [1.0, 1.0])
        try hnsw.insert(internalID: 1, vector: [2.0, 2.0])
        
        // getVector var
        let v0 = hnsw.getVector(internalID: 0)
        XCTAssertEqual(v0, [1.0, 1.0])
        
        // Tombstoned vector nil dönmeli
        try hnsw.remove(internalID: 0)
        let v0AfterDelete = hnsw.getVector(internalID: 0)
        XCTAssertNil(v0AfterDelete)
        
        let v1 = hnsw.getVector(internalID: 1)
        XCTAssertEqual(v1, [2.0, 2.0])
    }
}
