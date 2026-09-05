import Foundation

/// A minimal `.npy` reader and writer.
///
/// The bundle's arrays are `.npy` because that is what numpy writes without
/// ceremony and what a reader can parse in eighty lines. This supports exactly
/// what the format is used for here -- versions 1.0 and 2.0, C order, and the
/// four dtypes the bundle declares -- and refuses everything else with a typed
/// error naming what it found. A reader that silently reinterprets a Fortran
/// array as C order returns a transposed landscape; that is precisely the class
/// of failure a typed decode error exists to prevent.
public enum NPYError: Error, CustomStringConvertible, Equatable {
    case tooShort(String)
    case badMagic
    case unsupportedVersion(major: Int, minor: Int)
    case malformedHeader(String)
    case fortranOrder
    case unsupportedDType(String, expected: [String])
    case shapeMismatch(expected: [Int], found: [Int])
    case truncated(expectedBytes: Int, foundBytes: Int)

    public var description: String {
        switch self {
        case .tooShort(let what): "npy: too short to contain \(what)"
        case .badMagic: "npy: not a .npy file (bad magic)"
        case .unsupportedVersion(let a, let b): "npy: unsupported version \(a).\(b)"
        case .malformedHeader(let d): "npy: malformed header (\(d))"
        case .fortranOrder: "npy: Fortran-order arrays are not supported"
        case .unsupportedDType(let d, let e): "npy: dtype \(d) is not one of \(e)"
        case .shapeMismatch(let e, let f): "npy: expected shape \(e), found \(f)"
        case .truncated(let e, let f): "npy: expected \(e) bytes of data, found \(f)"
        }
    }
}

/// The dtypes a bundle may declare. Little-endian only: every consumer of this
/// format is Apple silicon or x86, and a big-endian file here would be a bug
/// rather than a portability need.
public enum NPYDType: String, Sendable, CaseIterable {
    case float32 = "<f4"
    case float64 = "<f8"
    case int64 = "<i8"
    case complex128 = "<c16"

    public var itemSize: Int {
        switch self {
        case .float32: 4
        case .float64: 8
        case .int64: 8
        case .complex128: 16
        }
    }
}

public struct NPYArray: Sendable {
    public let dtype: NPYDType
    public let shape: [Int]
    /// Raw little-endian element bytes, C order.
    public let data: [UInt8]

    public var count: Int { shape.reduce(1, *) }
}

public enum NPY {
    static let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]   // \x93NUMPY

    public static func read(contentsOf url: URL) throws -> NPYArray {
        try parse(Data(contentsOf: url))
    }

    public static func parse(_ data: Data) throws -> NPYArray {
        let bytes = [UInt8](data)
        guard bytes.count >= 10 else { throw NPYError.tooShort("a header") }
        guard Array(bytes[0..<6]) == magic else { throw NPYError.badMagic }

        let major = Int(bytes[6]), minor = Int(bytes[7])
        let headerLength: Int
        let headerStart: Int
        switch (major, minor) {
        case (1, 0):
            headerLength = Int(bytes[8]) | Int(bytes[9]) << 8
            headerStart = 10
        case (2, 0), (3, 0):
            guard bytes.count >= 12 else { throw NPYError.tooShort("a v2 header") }
            headerLength = Int(bytes[8]) | Int(bytes[9]) << 8
                | Int(bytes[10]) << 16 | Int(bytes[11]) << 24
            headerStart = 12
        default:
            throw NPYError.unsupportedVersion(major: major, minor: minor)
        }
        guard bytes.count >= headerStart + headerLength else {
            throw NPYError.tooShort("its declared header")
        }
        let header = String(decoding: bytes[headerStart..<(headerStart + headerLength)],
                            as: UTF8.self)

        let descr = try dictValue(header, "descr")
        let fortran = try dictValue(header, "fortran_order")
        let shapeText = try dictValue(header, "shape")

        if fortran == "True" { throw NPYError.fortranOrder }
        guard let dtype = NPYDType(rawValue: descr.trimmingCharacters(in: quotes)) else {
            throw NPYError.unsupportedDType(descr,
                                            expected: NPYDType.allCases.map(\.rawValue))
        }
        let shape = try parseShape(shapeText)

        let start = headerStart + headerLength
        let expected = shape.reduce(1, *) * dtype.itemSize
        let found = bytes.count - start
        guard found >= expected else {
            throw NPYError.truncated(expectedBytes: expected, foundBytes: found)
        }
        return NPYArray(dtype: dtype, shape: shape,
                        data: Array(bytes[start..<(start + expected)]))
    }

    private static let quotes = CharacterSet(charactersIn: "'\" ")

    /// The header is a Python dict literal. Rather than write a Python parser,
    /// find the key and take the balanced value that follows it -- the format
    /// guarantees the three keys and nothing nested beyond a tuple.
    private static func dictValue(_ header: String, _ key: String) throws -> String {
        guard let keyRange = header.range(of: "'\(key)'")
                ?? header.range(of: "\"\(key)\"") else {
            throw NPYError.malformedHeader("no key \(key)")
        }
        var i = header.index(after: keyRange.upperBound)   // past the ':'
        guard let colon = header[keyRange.upperBound...].firstIndex(of: ":") else {
            throw NPYError.malformedHeader("no ':' after \(key)")
        }
        i = header.index(after: colon)
        var depth = 0
        var out = ""
        while i < header.endIndex {
            let c = header[i]
            if c == "(" || c == "[" { depth += 1 }
            if c == ")" || c == "]" { depth -= 1 }
            if c == "," && depth == 0 { break }
            if c == "}" && depth == 0 { break }
            out.append(c)
            i = header.index(after: i)
        }
        return out.trimmingCharacters(in: .whitespaces)
    }

    private static func parseShape(_ text: String) throws -> [Int] {
        let inner = text.trimmingCharacters(in: CharacterSet(charactersIn: "() "))
        if inner.isEmpty { return [] }                       // 0-d
        var out: [Int] = []
        for piece in inner.split(separator: ",") {
            let t = piece.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { continue }
            guard let n = Int(t) else {
                throw NPYError.malformedHeader("shape component \(t)")
            }
            out.append(n)
        }
        return out
    }
}

// MARK: - typed views

public extension NPYArray {
    func floats() throws -> [Float] {
        guard dtype == .float32 else {
            throw NPYError.unsupportedDType(dtype.rawValue, expected: [NPYDType.float32.rawValue])
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    func doubles() throws -> [Double] {
        switch dtype {
        case .float64:
            return data.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
        case .float32:
            return try floats().map(Double.init)
        default:
            throw NPYError.unsupportedDType(
                dtype.rawValue,
                expected: [NPYDType.float64.rawValue, NPYDType.float32.rawValue])
        }
    }

    func ints() throws -> [Int] {
        guard dtype == .int64 else {
            throw NPYError.unsupportedDType(dtype.rawValue, expected: [NPYDType.int64.rawValue])
        }
        return data.withUnsafeBytes { $0.bindMemory(to: Int64.self).map(Int.init) }
    }

    /// Rows of an `(n, k)` array as `SIMD3` (k must be 3), the shape every
    /// vertex array in a bundle has.
    func rows3() throws -> [SIMD3<Double>] {
        guard shape.count == 2, shape[1] == 3 else {
            throw NPYError.shapeMismatch(expected: [-1, 3], found: shape)
        }
        let flat = try doubles()
        return (0..<shape[0]).map { SIMD3(flat[3 * $0], flat[3 * $0 + 1], flat[3 * $0 + 2]) }
    }

    func rows3i() throws -> [SIMD3<Int32>] {
        guard shape.count == 2, shape[1] == 3 else {
            throw NPYError.shapeMismatch(expected: [-1, 3], found: shape)
        }
        let flat = try ints()
        return (0..<shape[0]).map {
            SIMD3(Int32(flat[3 * $0]), Int32(flat[3 * $0 + 1]), Int32(flat[3 * $0 + 2]))
        }
    }
}

// MARK: - writing

public extension NPY {
    /// Write a C-order array. Only what the CLI needs to dump a depth image for
    /// comparison against the Python oracle -- the bundle itself is written by
    /// the Python side.
    static func write(_ values: [Float], shape: [Int], to url: URL) throws {
        precondition(shape.reduce(1, *) == values.count, "shape does not match count")
        var out = Data()
        let shapeText = shape.count == 1
            ? "(\(shape[0]),)"
            : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeText), }"
        // numpy pads the header so the data starts on a 64-byte boundary.
        let unpadded = 10 + header.utf8.count + 1
        let padding = (64 - unpadded % 64) % 64
        header += String(repeating: " ", count: padding) + "\n"

        out.append(contentsOf: NPY.magic)
        out.append(contentsOf: [1, 0])
        let n = header.utf8.count
        out.append(contentsOf: [UInt8(n & 0xFF), UInt8((n >> 8) & 0xFF)])
        out.append(contentsOf: Array(header.utf8))
        values.withUnsafeBufferPointer { out.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self)) }
        try out.write(to: url)
    }
}
