import Foundation

/// The expression language of `kurven/expr.py`, natively.
///
/// Same grammar, same six node kinds, same canonical spelling, same error
/// positions -- the fixtures in `tests/fixtures/expr` hold this parser to the
/// Python one on the ambiguous cases (`-z^2`, `2z^2`, `1/2z`) where a port
/// would otherwise silently pick the other reading. The function table is the
/// whitelist: a name that is not in it is not reachable.
///
///     expression := sum
///     sum        := product (("+" | "-") product)*
///     product    := unary (("*" | "/") unary | unary)*     (juxtaposition)
///     unary      := ("+" | "-") unary | power
///     power      := postfix (("^" | "**") unary)?          (right associative)
///     postfix    := atom "!"*                              (n! = Γ(n+1))
///     atom       := number | name | name "(" expression ("," expression)* ")"
///                 | "(" expression ")"
public struct ExpressionError: Error, CustomStringConvertible, Equatable, Sendable {
    public var message: String
    /// Character offset into the source, and how much of it is at fault, so a
    /// field can underline the mistake. `nil` position for empty input.
    public var position: Int?
    public var length: Int

    public init(_ message: String, at position: Int? = nil, length: Int = 1) {
        self.message = message; self.position = position; self.length = max(length, 1)
    }

    public var description: String {
        guard let position else { return message }
        return "\(message) (at character \(position + 1))"
    }
}

public enum Expression {
    // MARK: tokens

    enum TokenKind { case number, name, op, end }

    struct Token {
        var kind: TokenKind
        var text: String
        var position: Int
        var length: Int { max(text.count, 1) }
    }

    static let operators = ["**", "+", "-", "*", "/", "^", "!", "(", ")", ","]

    static func isNameStart(_ c: Character) -> Bool { c.isLetter || c == "_" }
    static func isNameChar(_ c: Character) -> Bool { c.isLetter || c.isNumber || c == "_" }
    static func isDigit(_ c: Character) -> Bool { c.isASCII && c.isNumber }

    static func tokenize(_ text: String) throws -> [Token] {
        let chars = Array(text)
        var tokens: [Token] = []
        var i = 0
        let n = chars.count
        while i < n {
            let c = chars[i]
            if c.isWhitespace { i += 1; continue }
            if isDigit(c) || (c == "." && i + 1 < n && isDigit(chars[i + 1])) {
                let start = i
                while i < n && isDigit(chars[i]) { i += 1 }
                if i < n && chars[i] == "." {
                    i += 1
                    while i < n && isDigit(chars[i]) { i += 1 }
                }
                // An `e` is an exponent only when digits follow it; otherwise
                // it is Euler's number and `2e` is a product.
                if i < n && (chars[i] == "e" || chars[i] == "E") {
                    var j = i + 1
                    if j < n && (chars[j] == "+" || chars[j] == "-") { j += 1 }
                    if j < n && isDigit(chars[j]) {
                        i = j
                        while i < n && isDigit(chars[i]) { i += 1 }
                    }
                }
                tokens.append(Token(kind: .number, text: String(chars[start..<i]), position: start))
                continue
            }
            if isNameStart(c) {
                let start = i
                while i < n && isNameChar(chars[i]) { i += 1 }
                tokens.append(Token(kind: .name, text: String(chars[start..<i]), position: start))
                continue
            }
            var matched = false
            for op in operators where text_hasPrefix(chars, i, op) {
                tokens.append(Token(kind: .op, text: op, position: i))
                i += op.count
                matched = true
                break
            }
            if !matched {
                throw ExpressionError("'\(c)' is not something this language uses", at: i)
            }
        }
        tokens.append(Token(kind: .end, text: "", position: n))
        return tokens
    }

    static func text_hasPrefix(_ chars: [Character], _ at: Int, _ prefix: String) -> Bool {
        let p = Array(prefix)
        guard at + p.count <= chars.count else { return false }
        for (k, c) in p.enumerated() where chars[at + k] != c { return false }
        return true
    }

    // MARK: the tree

    /// Precedence levels, for deciding where `format` must put parentheses.
    static let sum = 1, product = 2, unary = 3, power = 4, postfix = 5, atom = 6

    public indirect enum Node: Equatable, Sendable {
        case number(Double)
        case variable
        case constant(String, Complex)
        case call(String, [Node])
        case binary(String, Node, Node)
        case negate(Node)
        case factorial(Node)

        static func parenthesize(_ text: String, _ context: Int, _ mine: Int) -> String {
            mine < context ? "(\(text))" : text
        }

        /// Canonical source, the inverse of the parser.
        public func format() -> String { format(Expression.sum) }

        func format(_ context: Int) -> String {
            switch self {
            case .number(let v):
                if v == v.rounded() && abs(v) < 1e15 { return String(Int64(v)) }
                return formatDouble(v)
            case .variable: return "z"
            case .constant(let name, _): return name
            case .call(let name, let args):
                return "\(name)(\(args.map { $0.format() }.joined(separator: ", ")))"
            case .binary(let op, let l, let r):
                if op == "^" {
                    let text = "\(l.format(Expression.postfix))^\(r.format(Expression.unary))"
                    return Node.parenthesize(text, context, Expression.power)
                }
                let mine = (op == "+" || op == "-") ? Expression.sum : Expression.product
                let text = "\(l.format(mine))\(op)\(r.format(mine + 1))"
                return Node.parenthesize(text, context, mine)
            case .negate(let n):
                return Node.parenthesize("-\(n.format(Expression.unary))", context, Expression.unary)
            case .factorial(let n):
                return "\(n.format(Expression.postfix))!"
            }
        }

        /// The value at z. Total: a pole is inf and a branch point nan, never
        /// an error, because a landscape is a picture of where those are.
        public func evaluate(_ z: Complex) -> Complex {
            switch self {
            case .number(let v): return Complex(v)
            case .variable: return z
            case .constant(_, let v): return v
            case .call(let name, let args):
                return Language.function(name)!.apply(args.map { $0.evaluate(z) })
            case .binary(let op, let l, let r):
                let a = l.evaluate(z), b = r.evaluate(z)
                switch op {
                case "+": return a + b
                case "-": return a - b
                case "*": return a * b
                case "/": return a / b
                default: return Complex.pow(a, b)
                }
            case .negate(let n): return -n.evaluate(z)
            case .factorial(let n): return Gamma.gamma(n.evaluate(z) + 1)
            }
        }
    }

    /// Python's `repr(float)`: the shortest round-tripping decimal, with an
    /// exponent written as `e-05` rather than `e-5`.
    static func formatDouble(_ v: Double) -> String {
        var s = "\(v)"
        if let e = s.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            var mantissa = String(s[..<e])
            var exponent = String(s[s.index(after: e)...])
            var sign = "+"
            if exponent.hasPrefix("-") { sign = "-"; exponent.removeFirst() }
            else if exponent.hasPrefix("+") { exponent.removeFirst() }
            if exponent.count < 2 { exponent = "0" + exponent }
            if mantissa.hasSuffix(".0") { mantissa.removeLast(2) }
            s = "\(mantissa)e\(sign)\(exponent)"
        }
        return s
    }

    // MARK: the parser

    struct Parser {
        let text: String
        var tokens: [Token]
        var at = 0

        init(_ text: String) throws {
            self.text = text
            self.tokens = try Expression.tokenize(text)
        }

        var token: Token { tokens[at] }

        mutating func advance() -> Token {
            let t = tokens[at]
            at += 1
            return t
        }

        func lookingAt(_ texts: String...) -> Bool {
            token.kind == .op && texts.contains(token.text)
        }

        mutating func expect(_ text: String, _ what: String) throws -> Token {
            guard lookingAt(text) else {
                throw ExpressionError("expected '\(text)' \(what)", at: token.position,
                                      length: token.length)
            }
            return advance()
        }

        mutating func parse() throws -> Node {
            if token.kind == .end {
                throw ExpressionError("there is no expression here, only empty space", at: 0)
            }
            let node = try sum()
            if token.kind != .end {
                throw ExpressionError("'\(token.text)' has nothing to attach to",
                                      at: token.position, length: token.length)
            }
            return node
        }

        mutating func sum() throws -> Node {
            var node = try product()
            while lookingAt("+", "-") {
                let op = advance().text
                node = .binary(op, node, try product())
            }
            return node
        }

        mutating func product() throws -> Node {
            var node = try unary()
            while true {
                if lookingAt("*", "/") {
                    let op = advance().text
                    node = .binary(op, node, try unary())
                } else if startsAnAtom {
                    // Juxtaposition. The right side is a full `unary` so that
                    // `2z^2` is `2*(z^2)` rather than `(2*z)^2`.
                    node = .binary("*", node, try unary())
                } else {
                    return node
                }
            }
        }

        var startsAnAtom: Bool {
            token.kind == .number || token.kind == .name || (token.kind == .op && token.text == "(")
        }

        mutating func unary() throws -> Node {
            if lookingAt("-") { _ = advance(); return .negate(try unary()) }
            if lookingAt("+") { _ = advance(); return try unary() }
            return try power()
        }

        mutating func power() throws -> Node {
            let base = try postfix()
            if lookingAt("^", "**") {
                _ = advance()
                return .binary("^", base, try unary())
            }
            return base
        }

        mutating func postfix() throws -> Node {
            var node = try atom()
            while lookingAt("!") {
                _ = advance()
                node = .factorial(node)
            }
            return node
        }

        mutating func atom() throws -> Node {
            let t = token
            if t.kind == .number {
                _ = advance()
                return .number(Double(t.text) ?? .nan)
            }
            if t.kind == .name { return try name(advance()) }
            if lookingAt("(") {
                _ = advance()
                let node = try sum()
                _ = try expect(")", "to close the group")
                return node
            }
            if t.kind == .end {
                throw ExpressionError("the expression stops in the middle", at: t.position)
            }
            throw ExpressionError("'\(t.text)' cannot start a term", at: t.position, length: t.length)
        }

        mutating func name(_ t: Token) throws -> Node {
            guard let canonical = Language.canonicalNames[t.text] else {
                throw ExpressionError(Language.unknown(t.text), at: t.position, length: t.length)
            }
            if let f = Language.function(canonical) {
                guard lookingAt("(") else {
                    throw ExpressionError(
                        "\(canonical) is a function; it needs an argument, as in "
                        + "\(canonical)(\(f.arity == 2 ? "z, 0.5" : "z"))",
                        at: t.position, length: t.length)
                }
                _ = advance()
                var args = [try sum()]
                while lookingAt(",") {
                    _ = advance()
                    args.append(try sum())
                }
                let close = try expect(")", "to close the call to \(canonical)")
                if args.count != f.arity {
                    throw ExpressionError(
                        "\(canonical) takes \(f.arity) argument\(f.arity == 1 ? "" : "s"), "
                        + "not \(args.count)",
                        at: t.position, length: close.position - t.position + 1)
                }
                return .call(canonical, args)
            }
            if canonical == "z" { return .variable }
            return .constant(canonical, Language.constant(canonical)!.value)
        }
    }

    // MARK: the public three

    /// Text to tree, or an `ExpressionError` that says where.
    public static func parse(_ text: String) throws -> Node {
        var p = try Parser(text)
        return try p.parse()
    }

    /// The same expression, spelled one way: aliases resolved, implicit
    /// multiplication explicit, `**` as `^`, spacing gone. Idempotent.
    public static func canonical(_ text: String) throws -> String {
        try parse(text).format()
    }

    /// Text to a function of z.
    public static func compile(_ text: String) throws -> Compiled {
        Compiled(node: try parse(text))
    }

    public struct Compiled: Sendable {
        public let node: Node
        public var expression: String { node.format() }
        public func callAsFunction(_ z: Complex) -> Complex { node.evaluate(z) }
    }
}

/// The function table and the constants: everything the language can name.
///
/// This is the data `kurven.expr.LANGUAGE` reports over the service, kept in
/// step with it by the fixture. The help strings are the same words because a
/// picker shows them, whichever side is answering.
public enum Language {
    public struct Function: Sendable {
        public let name: String
        public let arity: Int
        public let aliases: [String]
        public let help: String
        /// What each argument is, as a control would name it: `argument` for
        /// a point of the plane, `order` or `modulus` for a parameter.
        public let parameters: [String]
        let apply: @Sendable ([Complex]) -> Complex
    }

    public struct Constant: Sendable {
        public let name: String
        public let value: Complex
        public let aliases: [String]
        public let help: String
    }

    /// A second argument that is a *parameter*, not a point of the plane: a
    /// Bessel order or an elliptic modulus. Its imaginary part is dropped,
    /// as `kurven.expr._parameter` drops it.
    static func parameter(_ v: Complex) -> Double { v.re }

    static func f1(_ name: String, _ help: String, _ aliases: [String] = [],
                   _ body: @escaping @Sendable (Complex) -> Complex) -> Function {
        Function(name: name, arity: 1, aliases: aliases, help: help,
                 parameters: ["argument"]) { body($0[0]) }
    }

    static func f2(_ name: String, _ help: String, _ aliases: [String] = [],
                   parameters: [String],
                   _ body: @escaping @Sendable (Complex, Complex) -> Complex) -> Function {
        Function(name: name, arity: 2, aliases: aliases, help: help,
                 parameters: parameters) { body($0[0], $0[1]) }
    }

    public static let functions: [Function] = [
        f1("exp", "e raised to z") { Complex.exp($0) },
        f1("log", "natural logarithm, principal branch", ["ln"]) { Complex.log($0) },
        f1("sqrt", "square root, principal branch") { Complex.sqrt($0) },
        f1("sin", "sine") { Complex.sin($0) },
        f1("cos", "cosine") { Complex.cos($0) },
        f1("tan", "tangent — poles at the odd multiples of π/2") { Complex.tan($0) },
        f1("cot", "cotangent") { 1 / Complex.tan($0) },
        f1("sec", "secant") { 1 / Complex.cos($0) },
        f1("csc", "cosecant") { 1 / Complex.sin($0) },
        f1("sinh", "hyperbolic sine") { Complex.sinh($0) },
        f1("cosh", "hyperbolic cosine") { Complex.cosh($0) },
        f1("tanh", "hyperbolic tangent") { Complex.tanh($0) },
        f1("asin", "inverse sine", ["arcsin"]) { Complex.asin($0) },
        f1("acos", "inverse cosine", ["arccos"]) { Complex.acos($0) },
        f1("atan", "inverse tangent", ["arctan"]) { Complex.atan($0) },
        f1("asinh", "inverse hyperbolic sine", ["arcsinh"]) { Complex.asinh($0) },
        f1("acosh", "inverse hyperbolic cosine", ["arccosh"]) { Complex.acosh($0) },
        f1("atanh", "inverse hyperbolic tangent", ["arctanh"]) { Complex.atanh($0) },
        f1("gamma", "Γ(z) — poles at the non-positive integers", ["Γ"]) { Gamma.gamma($0) },
        f1("rgamma", "1/Γ(z), entire; zeros at 0, -1, -2, ...") { Gamma.rgamma($0) },
        f1("loggamma", "log Γ(z), principal branch", ["lgamma"]) { Gamma.logGamma($0) },
        f1("digamma", "ψ(z) = Γ'(z)/Γ(z)", ["psi", "ψ"]) { Gamma.digamma($0) },
        f1("zeta", "ζ(z) — the pole is at z = 1", ["ζ"]) { Zeta.zeta($0) },
        f1("erf", "the error function") { Faddeeva.erf($0) },
        f1("erfc", "the complementary error function") { Faddeeva.erfc($0) },
        f1("erfi", "the imaginary error function") { Faddeeva.erfi($0) },
        f1("wofz", "the Faddeeva function w(z)", ["faddeeva"]) { Faddeeva.w($0) },
        f1("expi", "the exponential integral, continued as -E₁(-z)", ["Ei"]) { ExpInt.expi($0) },
        f1("lambertw", "W(z), the principal branch of the Lambert W function", ["W"]) {
            LambertW.w0($0)
        },
        f1("airyai", "Ai(z)", ["Ai"]) { Airy.ai($0) },
        f1("airybi", "Bi(z)", ["Bi"]) { Airy.bi($0) },
        f2("besselj", "J_n(z), first argument the order", ["jv"], parameters: ["order", "argument"]) { Bessel.j(parameter($0), $1) },
        f2("bessely", "Y_n(z), first argument the order", ["yv"], parameters: ["order", "argument"]) { Bessel.y(parameter($0), $1) },
        f2("besseli", "I_n(z), first argument the order", ["iv"], parameters: ["order", "argument"]) { Bessel.i(parameter($0), $1) },
        f2("besselk", "K_n(z), first argument the order", ["kv"], parameters: ["order", "argument"]) { Bessel.k(parameter($0), $1) },
        f2("sn", "Jacobi sn(z, m), doubly periodic", parameters: ["argument", "modulus"]) { Jacobi.complex($0, parameter($1)).sn },
        f2("cn", "Jacobi cn(z, m), doubly periodic", parameters: ["argument", "modulus"]) { Jacobi.complex($0, parameter($1)).cn },
        f2("dn", "Jacobi dn(z, m), doubly periodic", parameters: ["argument", "modulus"]) { Jacobi.complex($0, parameter($1)).dn },
        f1("abs", "|z|, as a real value") { Complex($0.magnitude) },
        f1("re", "the real part") { Complex($0.re) },
        f1("im", "the imaginary part") { Complex($0.im) },
        f1("conj", "the complex conjugate") { $0.conjugate },
        f1("arg", "the argument of z") { Complex($0.argument) },
    ]

    public static let constants: [Constant] = [
        Constant(name: "i", value: .i, aliases: ["j"], help: "the imaginary unit"),
        Constant(name: "pi", value: Complex(.pi), aliases: ["π"], help: "π"),
        Constant(name: "e", value: Complex(M_E), aliases: [], help: "Euler's number"),
        Constant(name: "tau", value: Complex(2 * .pi), aliases: ["τ"], help: "2π"),
    ]

    static let functionsByName: [String: Function] =
        Dictionary(uniqueKeysWithValues: functions.map { ($0.name, $0) })
    static let constantsByName: [String: Constant] =
        Dictionary(uniqueKeysWithValues: constants.map { ($0.name, $0) })

    /// Every spelling, mapped to the one the tree stores.
    public static let canonicalNames: [String: String] = {
        var names = ["z": "z"]
        for f in functions {
            names[f.name] = f.name
            for a in f.aliases { names[a] = f.name }
        }
        for c in constants {
            names[c.name] = c.name
            for a in c.aliases { names[a] = c.name }
        }
        return names
    }()

    public static func function(_ name: String) -> Function? { functionsByName[name] }
    public static func constant(_ name: String) -> Constant? { constantsByName[name] }

    /// The message for a name the language does not have, with the nearest
    /// names it does -- `difflib.get_close_matches` at a cutoff of 0.6, by
    /// the longest common subsequence rather than its matching blocks, which
    /// agrees on every name this short.
    static func unknown(_ text: String) -> String {
        let candidates = canonicalNames.keys.sorted()
        var scored: [(String, Double)] = []
        for c in candidates {
            let ratio = 2.0 * Double(lcs(Array(text), Array(c))) / Double(text.count + c.count)
            if ratio >= 0.6 { scored.append((c, ratio)) }
        }
        scored.sort { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
        let close = scored.prefix(3).map(\.0)
        if !close.isEmpty {
            return "there is no '\(text)'; did you mean \(close.joined(separator: " or "))?"
        }
        return "there is no '\(text)' in this language; the variable is z, and "
            + "`describe` lists what else there is"
    }

    static func lcs(_ a: [Character], _ b: [Character]) -> Int {
        guard !a.isEmpty, !b.isEmpty else { return 0 }
        var prev = [Int](repeating: 0, count: b.count + 1)
        var cur = prev
        for i in 1...a.count {
            for j in 1...b.count {
                cur[j] = a[i - 1] == b[j - 1] ? prev[j - 1] + 1 : max(prev[j], cur[j - 1])
            }
            swap(&prev, &cur)
        }
        return prev[b.count]
    }
}
