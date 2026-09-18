import XCTest

@testable import VectorDatabase

final class WALBackwardCompatibilityTests: XCTestCase {
    var fileURL: URL!

    override func setUp() {
        super.setUp()
        fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wal")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: fileURL)
        super.tearDown()
    }

    func testLegacyWALParsing() throws {
        // Construct a raw binary WAL file simulating the old format (opcode 0 and 1 only)
        // Record 1: insert (opcode 0), internalID = 42, ts = 1000, vector = [1.0, 2.0]
        let dimension = 2
        var data = Data()

        var op0: UInt8 = 0
        data.append(&op0, count: 1)

        var id1: Int32 = 42
        data.append(withUnsafeBytes(of: &id1) { Data($0) })

        var ts1: UInt64 = 1000
        data.append(withUnsafeBytes(of: &ts1) { Data($0) })

        let vec: [Float] = [1.0, 2.0]
        vec.withUnsafeBufferPointer { buf in
            data.append(buf)
        }

        // Record 2: delete (opcode 1), internalID = 42, ts = 1001
        var op1: UInt8 = 1
        data.append(&op1, count: 1)

        var id2: Int32 = 42
        data.append(withUnsafeBytes(of: &id2) { Data($0) })

        var ts2: UInt64 = 1001
        data.append(withUnsafeBytes(of: &ts2) { Data($0) })

        // Record 3: new insertWithMetadata (opcode 3), internalID = 43, ts = 1002, vector = [3.0, 4.0], metadata = {"k": "v"}
        var op3: UInt8 = 3
        data.append(&op3, count: 1)

        var id3: Int32 = 43
        data.append(withUnsafeBytes(of: &id3) { Data($0) })

        var ts3: UInt64 = 1002
        data.append(withUnsafeBytes(of: &ts3) { Data($0) })

        let vec3: [Float] = [3.0, 4.0]
        vec3.withUnsafeBufferPointer { buf in
            data.append(buf)
        }

        let metaJson = try JSONEncoder().encode(["k": "v"])
        var len = Int32(metaJson.count)
        data.append(withUnsafeBytes(of: &len) { Data($0) })
        data.append(metaJson)

        try data.write(to: fileURL)

        // Parse it with the new WriteAheadLog logic
        let wal = try WriteAheadLog(path: fileURL, dimension: dimension)

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
