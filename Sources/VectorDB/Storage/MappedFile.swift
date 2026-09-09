import Foundation
import Darwin

/// MappedFile.swift — Phase 6: Memory-mapped file wrapper.
///
/// Implements §10.3: `open`, `fstat`, `ftruncate`, `mmap`, `msync`, `munmap`.
public final class MappedFile {
    private var fd: Int32
    public private(set) var size: Int
    public private(set) var pointer: UnsafeMutableRawPointer

    public init(path: String, initialSize: Int) throws {
        // 1. Open or create the file
        fd = open(path, O_CREAT | O_RDWR, 0o666)
        guard fd != -1 else {
            throw VectorDBError.ioError(errno: errno)
        }

        // 2. Check current size
        var statBuf = stat()
        guard fstat(fd, &statBuf) == 0 else {
            close(fd)
            throw VectorDBError.ioError(errno: errno)
        }

        var currentSize = Int(statBuf.st_size)

        // 3. Resize if the file is smaller than required
        if currentSize < initialSize {
            guard ftruncate(fd, off_t(initialSize)) == 0 else {
                close(fd)
                throw VectorDBError.ioError(errno: errno)
            }
            currentSize = initialSize
        }

        self.size = currentSize

        // 4. Memory map the file
        let mapped = mmap(nil, currentSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard mapped != MAP_FAILED else {
            close(fd)
            throw VectorDBError.ioError(errno: errno)
        }

        self.pointer = mapped!
    }

    deinit {
        munmap(pointer, size)
        close(fd)
    }

    /// Resizes the file on disk and remaps it into memory.
    ///
    /// DARWIN GOTCHA: Darwin does not have `mremap`. We must explicitly `munmap`,
    /// `ftruncate`, and `mmap` again. This means the newly mapped region may have
    /// a DIFFERENT base virtual address. Callers MUST NOT cache `pointer` across
    /// this call.
    public func resize(to newSize: Int) throws {
        guard newSize > size else { return }

        // Unmap the old region
        guard munmap(pointer, size) == 0 else {
            throw VectorDBError.ioError(errno: errno)
        }

        // Grow the file on disk
        guard ftruncate(fd, off_t(newSize)) == 0 else {
            throw VectorDBError.ioError(errno: errno)
        }

        // Map the new region
        let mapped = mmap(nil, newSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard mapped != MAP_FAILED else {
            throw VectorDBError.ioError(errno: errno)
        }

        self.pointer = mapped!
        self.size = newSize
    }

    /// Flushes all dirty mapped pages to disk.
    public func sync() throws {
        guard msync(pointer, size, MS_SYNC) == 0 else {
            throw VectorDBError.ioError(errno: errno)
        }
    }
}
