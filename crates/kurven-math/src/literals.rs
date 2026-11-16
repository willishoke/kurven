//! The numbers written in an expression, as things that can be moved. The
//! port of `Literals.swift`.
//!
//! `cn(z, 0.64)` has a modulus in it, `besselj(2, z)` an order, `z^3` an
//! exponent -- and none of them has a name. A parameter language would give
//! them names and a scope; this gives them a *place*: every number literal in
//! the source, with its character range, so a slider can rewrite that range
//! and the expression stays the only state there is. What the text says is
//! what the landscape is, after a drag as before it.

use crate::expression::{Expression, Token, TokenKind, format_number_plain, tokenize};
use crate::language::Language;

#[derive(Clone, PartialEq, Debug)]
pub struct Literal {
    /// Character offset of the first character, the sign when there is one.
    pub position: usize,
    pub length: usize,
    /// The number as written, sign included.
    pub value: f64,
    /// The source characters, `-0.5` or `1e-3`.
    pub text: String,
    /// Whether a bare `-` here would read as this number's sign: true at
    /// the start or after an operator, false after an operand (where it
    /// would be subtraction) or before a `!` (which binds tighter).
    pub signable: bool,
}

impl Literal {
    /// A literal, signable unless said otherwise (Swift's default argument).
    pub fn new(position: usize, length: usize, value: f64, text: &str) -> Literal {
        Literal { position, length, value, text: text.to_string(), signable: true }
    }

    pub fn end(&self) -> usize {
        self.position + self.length
    }
}

/// What a number is doing where it is, as a control would label it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Role {
    /// The right side of `^`.
    Exponent,
    /// A factor: next to `*`, before a name or a group, or over a `/`.
    Coefficient,
    /// Under a `/`.
    Divisor,
    /// The `index`th argument of a call, named as the function names it.
    Argument { function: String, index: usize, parameter: String },
    /// A term of a sum, or the whole expression.
    Constant,
}

impl Role {
    /// The label a control gets: "Modulus", "Order", "Exponent".
    pub fn label(&self) -> String {
        match self {
            Role::Exponent => "Exponent".to_string(),
            Role::Coefficient => "Coefficient".to_string(),
            Role::Divisor => "Divisor".to_string(),
            Role::Argument { parameter, .. } => capitalized(parameter),
            Role::Constant => "Constant".to_string(),
        }
    }
}

/// Foundation's `capitalized` on the one-word parameter names: the first
/// letter upper case, the rest lower.
fn capitalized(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars.flat_map(char::to_lowercase)).collect(),
        None => String::new(),
    }
}

impl Expression {
    /// Every number literal in `text`, in source order.
    ///
    /// A minus sign directly before a number is part of the literal when it is
    /// *unary* -- at the start, after an operator, after `(` or `,` -- so that
    /// `2^-0.5` offers a slider on `-0.5` that can cross zero. After an operand
    /// (`z-0.5`) the minus is subtraction and the literal is `0.5`. Text the
    /// tokenizer refuses has no literals.
    pub fn literals(text: &str) -> Vec<Literal> {
        let Ok(tokens) = tokenize(text) else { return Vec::new() };
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        for (k, t) in tokens.iter().enumerate() {
            if t.kind != TokenKind::Number {
                continue;
            }
            let Ok(v) = t.text.parse::<f64>() else { continue };
            if !v.is_finite() {
                continue;
            }
            let mut position = t.position;
            let mut value = v;
            // A sign written here is the number's own only after an operator
            // that is not itself a sign: `2^-0.5`, `cn(z, -0.5)`, not `z--0.5`.
            let mut signable =
                k == 0 || (tokens[k - 1].kind == TokenKind::Op && !tokens[k - 1].is_op(&["+", "-", ")", "!"]));
            if k > 0 && tokens[k - 1].is_op(&["-"]) && minus_is_unary(&tokens, k - 1) {
                position = tokens[k - 1].position;
                value = -v;
                signable = true;
            }
            if tokens[k + 1].is_op(&["!"]) {
                signable = false;
            }
            let end = t.position + t.count();
            out.push(Literal {
                position,
                length: end - position,
                value,
                text: chars[position..end].iter().collect(),
                signable,
            });
        }
        out
    }

    /// `text` with one literal rewritten to `value`.
    ///
    /// A negative value is parenthesized where a bare sign would not be the
    /// number's own -- `z-0.5` becomes `z-(-0.3)`, `2!` becomes `(-1)!`,
    /// because `-1!` would be `-(1!)` -- and written bare where it would,
    /// `2^0.25` to `2^-0.5`. A slider must not change the shape of the
    /// expression it moves through. The literal is then found again, sign
    /// and all, at the same index.
    pub fn replacing(literal: &Literal, value: f64, significant: Option<usize>, text: &str) -> String {
        let mut chars: Vec<char> = text.chars().collect();
        if literal.end() > chars.len() {
            return text.to_string();
        }
        let mut replacement = Expression::format_number(value, significant);
        if value < 0.0 && !literal.signable {
            replacement = format!("({replacement})");
        }
        chars.splice(literal.position..literal.end(), replacement.chars());
        chars.into_iter().collect()
    }

    /// A number as the language writes it: an integer without a point, and
    /// otherwise the shortest decimal that reads back as the same double.
    /// With `significant`, rounded to that many digits first, so a value off a
    /// slider is `0.3` rather than `0.30000000000000004`.
    pub fn format_number(value: f64, significant: Option<usize>) -> String {
        let mut v = value;
        if let Some(significant) = significant
            && v.is_finite()
        {
            // C's `%.{n}g`, which Swift calls, rounds to n significant digits
            // exactly; so does `{:.{n-1}e}`, and the layout does not matter to
            // the double read back.
            v = format!("{:.*e}", significant.max(1) - 1, v).parse().unwrap_or(v);
        }
        format_number_plain(v)
    }

    /// The role of one literal, read from the tokens around it.
    ///
    /// Checked in the order that decides: the `^` before it makes it an
    /// exponent even inside a call; a `*`, `/` or juxtaposition beside it
    /// makes it a factor even inside a call; only then does an enclosing
    /// call name it after the parameter it fills.
    pub fn role(literal: &Literal, text: &str) -> Role {
        let Ok(tokens) = tokenize(text) else { return Role::Constant };
        let Some(k) =
            tokens.iter().position(|t| t.kind == TokenKind::Number && t.position + t.count() == literal.end())
        else {
            return Role::Constant;
        };
        // The token before the number, skipping the sign that is the literal's own.
        let mut before = k as isize - 1;
        if before >= 0 && tokens[before as usize].position == literal.position && literal.text.starts_with('-') {
            before -= 1;
        }
        let prev: Option<&Token> = if before >= 0 { Some(&tokens[before as usize]) } else { None };
        let next = &tokens[k + 1];
        let is_op = |t: Option<&Token>, texts: &[&str]| t.is_some_and(|t| t.is_op(texts));
        if is_op(prev, &["^", "**"]) {
            return Role::Exponent;
        }
        if is_op(prev, &["/"]) {
            return Role::Divisor;
        }
        if is_op(prev, &["*"]) || next.is_op(&["*", "/"]) || next.kind == TokenKind::Name || next.is_op(&["("]) {
            return Role::Coefficient;
        }
        // Walk back to the `(` that encloses this literal, counting the commas
        // at its level, to find which argument of which call this is.
        let mut depth = 0;
        let mut index = 0;
        let mut i = before;
        while i >= 0 {
            let t = &tokens[i as usize];
            if t.kind == TokenKind::Op {
                if t.text == ")" {
                    depth += 1;
                } else if t.text == "(" {
                    if depth == 0 {
                        if i > 0
                            && tokens[i as usize - 1].kind == TokenKind::Name
                            && let Some(name) = Language::canonical_name(&tokens[i as usize - 1].text)
                            && let Some(f) = Language::function(name)
                        {
                            let parameter = f.parameters.get(index).copied().unwrap_or("argument");
                            return Role::Argument {
                                function: name.to_string(),
                                index,
                                parameter: parameter.to_string(),
                            };
                        }
                        return Role::Constant;
                    }
                    depth -= 1;
                } else if t.text == "," && depth == 0 {
                    index += 1;
                }
            }
            i -= 1;
        }
        Role::Constant
    }
}

/// A `-` at `k` is unary unless something it could subtract from precedes it.
fn minus_is_unary(tokens: &[Token], k: usize) -> bool {
    if k == 0 {
        return true;
    }
    let p = &tokens[k - 1];
    if p.kind == TokenKind::Number || p.kind == TokenKind::Name {
        return false;
    }
    if p.is_op(&[")", "!"]) {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::expression::Node;

    /// `literalTests` in `Sources/kurven-test/Expressions.swift`.
    #[test]
    fn every_number_in_the_source_with_its_place() {
        let one = Expression::literals("sin(z)cn(z, 0.64)");
        assert_eq!(one, [Literal::new(12, 4, 0.64, "0.64")], "cn(z, 0.64) has one literal, the modulus");

        let mixed = Expression::literals("-z^2 + 2^-0.5 z - 3");
        assert_eq!(mixed.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), ["2", "2", "-0.5", "3"]);
        assert_eq!(mixed.iter().map(|l| l.value).collect::<Vec<_>>(), [2.0, 2.0, -0.5, 3.0]);
        assert_eq!(mixed.iter().map(|l| l.position).collect::<Vec<_>>(), [3, 7, 9, 18]);

        let forms = Expression::literals("1e-3 z! + (-1)! + .5");
        assert_eq!(forms.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), ["1e-3", "-1", ".5"]);
        assert_eq!(forms.iter().map(|l| l.value).collect::<Vec<_>>(), [1e-3, -1.0, 0.5]);

        assert!(Expression::literals("gamma(z)").is_empty(), "an expression with no numbers has none");
        assert!(Expression::literals("2 $ z").is_empty(), "text the tokenizer refuses has none");
    }

    #[test]
    fn rewriting_one_keeps_the_expressions_shape() {
        fn moved(text: &str, index: usize, value: f64) -> String {
            Expression::replacing(&Expression::literals(text)[index], value, None, text)
        }
        assert_eq!(moved("sin(z)cn(z, 0.64)", 0, 0.71), "sin(z)cn(z, 0.71)");
        assert_eq!(moved("2^-0.5", 1, 0.25), "2^0.25", "a signed literal crossing zero drops its sign");
        assert_eq!(moved("2^0.25", 1, -0.5), "2^-0.5", "an unsigned one after an operator gains one");
        assert_eq!(moved("2!", 0, -1.0), "(-1)!");
        assert_eq!(moved("(-1)!", 0, 2.0), "(2)!");
        assert_eq!(moved("z-0.5", 0, -0.3), "z-(-0.3)");
        let parsed = Expression::parse("(-1)!").unwrap();
        assert_eq!(parsed, Node::Factorial(Box::new(Node::Negate(Box::new(Node::Number(1.0))))));
        let again = Expression::literals(&moved("z-0.5", 0, -0.3));
        assert!(again.len() == 1 && again[0].value == -0.3 && again[0].text == "-0.3", "{again:?}");
        assert_eq!(moved("2z + 3", 0, 2.5), "2.5z + 3");
        assert_eq!(moved("2exp(z)", 0, 2.5), "2.5exp(z)");
        let f = Expression::compile(&moved("z - 0.5", 0, -0.3)).unwrap();
        assert!((f.call(Complex::real(1.0)) - Complex::real(1.3)).magnitude() < 1e-15);
    }

    #[test]
    fn numbers_are_written_the_way_the_language_writes_them() {
        assert_eq!(Expression::format_number(2.0, None), "2");
        assert_eq!(Expression::format_number(-3.0, None), "-3");
        assert_eq!(Expression::format_number(0.1 + 0.2, Some(6)), "0.3");
        assert_eq!(Expression::format_number(1e-5, None), "1e-05");
        assert_eq!(Expression::format_number(0.64, None), "0.64");
        // C's %g, as Swift's String(format:) writes it on this machine.
        assert_eq!(Expression::format_number(1234567.0, Some(3)), "1230000");
        assert_eq!(Expression::format_number(0.125, Some(2)), "0.12");
        assert_eq!(Expression::format_number(0.25, Some(1)), "0.2");
        assert_eq!(Expression::format_number(0.35, Some(1)), "0.3");
    }

    #[test]
    fn each_numbers_role_is_what_a_control_would_call_it() {
        fn roles(text: &str) -> Vec<String> {
            Expression::literals(text).iter().map(|l| Expression::role(l, text).label()).collect()
        }
        assert_eq!(roles("sin(z)cn(z, 0.64)"), ["Modulus"]);
        assert_eq!(roles("besselj(2, z)"), ["Order"]);
        assert_eq!(roles("gamma(2)"), ["Argument"]);
        assert_eq!(roles("z^3 - 1"), ["Exponent", "Constant"]);
        assert_eq!(roles("2z + 1/z + z/2"), ["Coefficient", "Coefficient", "Divisor"]);
        assert_eq!(roles("cn(2z, -0.5)^-2"), ["Coefficient", "Modulus", "Exponent"]);
        assert_eq!(roles("(z+1)(z-2)"), ["Constant", "Constant"]);
        assert_eq!(Language::function("cn").unwrap().parameters, ["argument", "modulus"]);
        assert_eq!(Language::function("besselk").unwrap().parameters, ["order", "argument"]);
        assert_eq!(Language::function("exp").unwrap().parameters, ["argument"]);
    }
}
