/// VectorDBError.swift — Complete public error taxonomy (§12).
/// Defined in Phase 1 so FlatIndex and all later phases share the same types.
/// Cases are added to the public surface only; their values match the guide exactly.
public enum VectorDBError: Error, Sendable {
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
}
