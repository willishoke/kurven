import Foundation

/// The numbers written in an expression, as things that can be moved.
///
/// `cn(z, 0.64)` has a modulus in it, `besselj(2, z)` an order, `z^3` an
/// exponent -- and none of them has a name. A parameter language would give
/// them names and a scope; this gives them a *place*: every number literal in
/// the source, with its character range, so a slider can rewrite that range
/// and the expression stays the only state there is. What the text says is
/// what the landscape is, after a drag as before it.
extension Expression {
    public struct Literal: Equatable, Sendable {
        /// Character offset of the first character, the sign when there is one.
        public var position: Int
        public var length: Int
        /// The number as written, sign included.
        public var value: Double
        /// The source characters, `-0.5` or `1e-3`.
        public var text: String
        /// Whether a bare `-` here would read as this number's sign: true at
        /// the start or after an operator, false after an operand (where it
        /// would be subtraction) or before a `!` (which binds tighter).
        public var signable: Bool

        public var end: Int { position + length }

        public init(position: Int, length: Int, value: Double, text: String,
                    signable: Bool = true) {
            self.position = position; self.length = length; self.value = value; self.text = text
            self.signable = signable
        }
    }

    /// Every number literal in `text`, in source order.
    ///
    /// A minus sign directly before a number is part of the literal when it is
    /// *unary* -- at the start, after an operator, after `(` or `,` -- so that
    /// `2^-0.5` offers a slider on `-0.5` that can cross zero. After an operand
    /// (`z-0.5`) the minus is subtraction and the literal is `0.5`. Text the
    /// tokenizer refuses has no literals.
    public static func literals(in text: String) -> [Literal] {
        guard let tokens = try? tokenize(text) else { return [] }
        let chars = Array(text)
        var out: [Literal] = []
        for (k, t) in tokens.enumerated() where t.kind == .number {
            guard let v = Double(t.text), v.isFinite else { continue }
            var position = t.position
            var value = v
            // A sign written here is the number's own only after an operator
            // that is not itself a sign: `2^-0.5`, `cn(z, -0.5)`, not `z--0.5`.
            var signable = k == 0 || (tokens[k - 1].kind == .op
                                      && !["+", "-", ")", "!"].contains(tokens[k - 1].text))
            if k > 0, tokens[k - 1].kind == .op, tokens[k - 1].text == "-",
               minusIsUnary(tokens, at: k - 1) {
                position = tokens[k - 1].position
                value = -v
                signable = true
            }
            if tokens[k + 1].kind == .op && tokens[k + 1].text == "!" { signable = false }
            let end = t.position + t.text.count
            out.append(Literal(position: position, length: end - position, value: value,
                               text: String(chars[position..<end]), signable: signable))
        }
        return out
    }

    /// A `-` at `k` is unary unless something it could subtract from precedes it.
    static func minusIsUnary(_ tokens: [Token], at k: Int) -> Bool {
        guard k > 0 else { return true }
        let p = tokens[k - 1]
        if p.kind == .number || p.kind == .name { return false }
        if p.kind == .op && (p.text == ")" || p.text == "!") { return false }
        return true
    }

    /// `text` with one literal rewritten to `value`.
    ///
    /// A negative value is parenthesized where a bare sign would not be the
    /// number's own -- `z-0.5` becomes `z-(-0.3)`, `2!` becomes `(-1)!`,
    /// because `-1!` would be `-(1!)` -- and written bare where it would,
    /// `2^0.25` to `2^-0.5`. A slider must not change the shape of the
    /// expression it moves through. The literal is then found again, sign
    /// and all, at the same index.
    public static func replacing(_ literal: Literal, with value: Double, significant: Int? = nil,
                                 in text: String) -> String {
        var chars = Array(text)
        guard literal.position >= 0, literal.end <= chars.count else { return text }
        var replacement = formatNumber(value, significant: significant)
        if value < 0 && !literal.signable {
            replacement = "(\(replacement))"
        }
        chars.replaceSubrange(literal.position..<literal.end, with: Array(replacement))
        return String(chars)
    }

    /// A number as the language writes it: an integer without a point, and
    /// otherwise the shortest decimal that reads back as the same double.
    /// With `significant`, rounded to that many digits first, so a value off a
    /// slider is `0.3` rather than `0.30000000000000004`.
    public static func formatNumber(_ value: Double, significant: Int? = nil) -> String {
        var v = value
        if let significant, v.isFinite {
            v = Double(String(format: "%.\(max(significant, 1))g", v)) ?? v
        }
        if v == v.rounded() && abs(v) < 1e15 { return String(Int64(v)) }
        return formatDouble(v)
    }

    /// What a number is doing where it is, as a control would label it.
    public enum Role: Equatable, Sendable {
        /// The right side of `^`.
        case exponent
        /// A factor: next to `*`, before a name or a group, or over a `/`.
        case coefficient
        /// Under a `/`.
        case divisor
        /// The `index`th argument of a call, named as the function names it.
        case argument(function: String, index: Int, parameter: String)
        /// A term of a sum, or the whole expression.
        case constant

        /// The label a control gets: "Modulus", "Order", "Exponent".
        public var label: String {
            switch self {
            case .exponent: return "Exponent"
            case .coefficient: return "Coefficient"
            case .divisor: return "Divisor"
            case .argument(_, _, let parameter): return parameter.capitalized
            case .constant: return "Constant"
            }
        }
    }

    /// The role of one literal, read from the tokens around it.
    ///
    /// Checked in the order that decides: the `^` before it makes it an
    /// exponent even inside a call; a `*`, `/` or juxtaposition beside it
    /// makes it a factor even inside a call; only then does an enclosing
    /// call name it after the parameter it fills.
    public static func role(of literal: Literal, in text: String) -> Role {
        guard let tokens = try? tokenize(text),
              let k = tokens.firstIndex(where: {
                  $0.kind == .number && $0.position + $0.text.count == literal.end
              })
        else { return .constant }
        // The token before the number, skipping the sign that is the literal's own.
        var before = k - 1
        if before >= 0, tokens[before].position == literal.position, literal.text.hasPrefix("-") {
            before -= 1
        }
        let prev = before >= 0 ? tokens[before] : nil
        let next = tokens[k + 1]
        func isOp(_ t: Token?, _ texts: String...) -> Bool {
            guard let t, t.kind == .op else { return false }
            return texts.contains(t.text)
        }
        if isOp(prev, "^", "**") { return .exponent }
        if isOp(prev, "/") { return .divisor }
        if isOp(prev, "*") || isOp(next, "*", "/") || next.kind == .name || isOp(next, "(") {
            return .coefficient
        }
        // Walk back to the `(` that encloses this literal, counting the commas
        // at its level, to find which argument of which call this is.
        var depth = 0
        var index = 0
        var i = before
        while i >= 0 {
            let t = tokens[i]
            if t.kind == .op {
                if t.text == ")" { depth += 1 }
                else if t.text == "(" {
                    if depth == 0 {
                        if i > 0, tokens[i - 1].kind == .name,
                           let name = Language.canonicalNames[tokens[i - 1].text],
                           let f = Language.function(name) {
                            let parameter = index < f.parameters.count
                                ? f.parameters[index] : "argument"
                            return .argument(function: name, index: index, parameter: parameter)
                        }
                        return .constant
                    }
                    depth -= 1
                } else if t.text == "," && depth == 0 {
                    index += 1
                }
            }
            i -= 1
        }
        return .constant
    }
}
