import Darwin
import Foundation

/// VectorDatabaseError.swift — Complete public error taxonomy (§12).
/// Defined in Phase 1 so FlatIndex and all later phases share the same types.
/// Cases are added to the public surface only; their values match the guide exactly.
public enum VectorDatabaseError: Error, Sendable {
    /// The inserted/queried vector's dimension ≠ the configured dimension.
    /// Always validated at the public API layer before reaching unsafe pointer code.
    case dimensionMismatch(expected: Int, got: Int)

    /// insert() called for an ID that already exists in the index.
    /// Use update() for intentional overwrites.
    case duplicateID(String)

    /// delete() / update() called for an ID that does not exist.
    case notFound(String)

    /// Vector contains NaN, Inf, or (for .cosine) is a zero vector.
    case invalidVector(reason: String)

    /// A Darwin syscall (open, mmap, ftruncate, msync, …) failed.
    case ioError(errno: Int32)

    /// The file's magic bytes are wrong, or the body checksum fails.
    case corruptFile(reason: String)

    /// The file's formatVersion is newer than this library understands.
    case unsupportedFileVersion(found: UInt32, supported: UInt32)

    /// An HNSW parameter value is out of the valid range.
    case invalidParameters(reason: String)

    /// A legacy WAL format was encountered which does not contain the required fields (e.g., externalID).
    case legacyWALFormatNotSupported(reason: String)
}

/// `LocalizedError` conformance so a UI layer can show `error.localizedDescription`
/// (or `errorDescription`) directly in an alert, instead of the raw enum dump
/// (e.g. `dimensionMismatch(expected: 384, got: 128)`) that `String(describing:)`
/// would otherwise produce.
extension VectorDatabaseError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let expected, let got):
            return "This vector has \(got) dimension(s), but the database expects \(expected)."

        case .duplicateID(let id):
            return "An entry with ID \"\(id)\" already exists. Use update() to overwrite it."

        case .notFound(let id):
            return "No entry with ID \"\(id)\" was found."

        case .invalidVector(let reason):
            return "The vector is invalid: \(reason)"

        case .ioError(let errno):
            let message = String(cString: strerror(errno))
            return "A file system error occurred (errno \(errno): \(message))."

        case .corruptFile(let reason):
            return "The database file appears to be corrupted: \(reason)"

        case .unsupportedFileVersion(let found, let supported):
            return "This file was written by a newer, incompatible version of VectorDatabase "
                + "(format \(found)); this library supports up to format \(supported)."

        case .invalidParameters(let reason):
            return "Invalid configuration parameters: \(reason)"

        case .legacyWALFormatNotSupported(let reason):
            return "This write-ahead log uses an unsupported legacy format: \(reason)"
        }
    }
}
