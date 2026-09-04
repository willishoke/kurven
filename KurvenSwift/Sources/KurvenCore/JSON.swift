import Foundation

/// A JSON value, and a canonical writer for it.
///
/// `Codable` is deliberately not used for the manifest. The contract test is
/// "decode a manifest, re-encode it, compare bytes with what Python wrote", and
/// `JSONEncoder` cannot produce Python-canonical output: it renders the `Double`
/// 5.0 as `5`, so a cap height would round-trip to a different document every
/// time. Owning the writer costs sixty lines and buys an exact comparison, plus
/// a decoder whose errors name the key that was wrong instead of a coding path.
///
/// The distinction between `.int` and `.double` is meaningful here for the same
/// reason: the Python side writes `int()` for counts and `float()` for
/// measurements, and the canonical form must agree.
public enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}

public enum JSONError: Error, CustomStringConvertible, Equatable {
    case notAnObject(String)
    case missingKey(String, in: String)
    case wrongType(String, expected: String, found: String)
    case badSyntax(String)

    public var description: String {
        switch self {
        case .notAnObject(let w): "json: \(w) is not an object"
        case .missingKey(let k, let w): "json: \(w) has no key '\(k)'"
        case .wrongType(let k, let e, let f): "json: '\(k)' in \(f) should be \(e)"
        case .badSyntax(let d): "json: \(d)"
        }
    }
}

public extension JSONValue {
    static func parse(_ data: Data) throws -> JSONValue {
        let any = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        return JSONValue(any)
    }

    static func parse(_ text: String) throws -> JSONValue {
        try parse(Data(text.utf8))
    }

    init(_ any: Any) {
        switch any {
        case is NSNull: self = .null
        case let n as NSNumber:
            // NSNumber does not remember whether JSON said 5 or 5.0, but it does
            // remember the C type it boxed: JSONSerialization boxes an integral
            // literal as a long long and a fractional one as a double.
            let type = String(cString: n.objCType)
            if type == "c" || type == "B" { self = .bool(n.boolValue) }
            else if type == "d" || type == "f" { self = .double(n.doubleValue) }
            else { self = .int(n.intValue) }
        case let s as String: self = .string(s)
        case let a as [Any]: self = .array(a.map(JSONValue.init))
        case let o as [String: Any]: self = .object(o.mapValues(JSONValue.init))
        default: self = .null
        }
    }

    /// Canonical form: sorted keys, no insignificant whitespace, and numbers
    /// formatted as Python's `json` formats them. Two documents are equal iff
    /// their canonical forms are byte-equal.
    var canonical: String {
        switch self {
        case .null: "null"
        case .bool(let b): b ? "true" : "false"
        case .int(let i): String(i)
        case .double(let d): JSONValue.number(d)
        case .string(let s): JSONValue.quote(s)
        case .array(let a): "[" + a.map(\.canonical).joined(separator: ",") + "]"
        case .object(let o):
            "{" + o.keys.sorted().map { JSONValue.quote($0) + ":" + o[$0]!.canonical }
                .joined(separator: ",") + "}"
        }
    }

    /// Python's `repr` for a float: the shortest decimal that round-trips, with
    /// a trailing `.0` on integral values. Swift's `Double.description` agrees,
    /// so this is mostly a place to state the requirement and to refuse the
    /// values JSON cannot represent rather than emit an unparseable `inf`.
    static func number(_ d: Double) -> String {
        precondition(d.isFinite, "JSON cannot represent \(d); clamp before encoding")
        return d.description
    }

    static func quote(_ s: String) -> String {
        var out = "\""
        for c in s.unicodeScalars {
            switch c {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if c.value < 0x20 {
                    out += String(format: "\\u%04x", c.value)
                } else {
                    out.unicodeScalars.append(c)
                }
            }
        }
        return out + "\""
    }
}

// MARK: - typed access

public extension JSONValue {
    func object(_ what: String) throws -> [String: JSONValue] {
        guard case .object(let o) = self else { throw JSONError.notAnObject(what) }
        return o
    }
}

public extension Dictionary where Key == String, Value == JSONValue {
    func value(_ key: String, _ what: String) throws -> JSONValue {
        guard let v = self[key] else { throw JSONError.missingKey(key, in: what) }
        return v
    }

    func string(_ key: String, _ what: String) throws -> String {
        guard case .string(let s) = try value(key, what) else {
            throw JSONError.wrongType(key, expected: "a string", found: what)
        }
        return s
    }

    func int(_ key: String, _ what: String) throws -> Int {
        switch try value(key, what) {
        case .int(let i): return i
        case .double(let d) where d == d.rounded(): return Int(d)
        default: throw JSONError.wrongType(key, expected: "an integer", found: what)
        }
    }

    func double(_ key: String, _ what: String) throws -> Double {
        switch try value(key, what) {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: throw JSONError.wrongType(key, expected: "a number", found: what)
        }
    }

    func optionalDouble(_ key: String) -> Double? {
        switch self[key] {
        case .some(.double(let d)): return d
        case .some(.int(let i)): return Double(i)
        default: return nil
        }
    }

    func bool(_ key: String, _ what: String, default fallback: Bool? = nil) throws -> Bool {
        guard let v = self[key] else {
            if let fallback { return fallback }
            throw JSONError.missingKey(key, in: what)
        }
        guard case .bool(let b) = v else {
            throw JSONError.wrongType(key, expected: "a boolean", found: what)
        }
        return b
    }

    func array(_ key: String, _ what: String) throws -> [JSONValue] {
        guard case .array(let a) = try value(key, what) else {
            throw JSONError.wrongType(key, expected: "an array", found: what)
        }
        return a
    }

    func doubles(_ key: String, _ what: String) throws -> [Double] {
        try array(key, what).map {
            switch $0 {
            case .double(let d): return d
            case .int(let i): return Double(i)
            default: throw JSONError.wrongType(key, expected: "an array of numbers", found: what)
            }
        }
    }

    func ints(_ key: String, _ what: String) throws -> [Int] {
        try array(key, what).map {
            switch $0 {
            case .int(let i): return i
            case .double(let d) where d == d.rounded(): return Int(d)
            default: throw JSONError.wrongType(key, expected: "an array of integers", found: what)
            }
        }
    }

    func strings(_ key: String, _ what: String) throws -> [String] {
        try array(key, what).map {
            guard case .string(let s) = $0 else {
                throw JSONError.wrongType(key, expected: "an array of strings", found: what)
            }
            return s
        }
    }
}
