//! The expression language of `kurven/expr.py`, natively. The port of
//! `Expression.swift`.
//!
//! Same grammar, same six node kinds, same canonical spelling, same error
//! positions -- the fixtures in `tests/fixtures/expr` hold this parser to the
//! Python one on the ambiguous cases (`-z^2`, `2z^2`, `1/2z`) where a port
//! would otherwise silently pick the other reading. The function table is the
//! whitelist: a name that is not in it is not reachable.
//!
//! ```text
//! expression := sum
//! sum        := product (("+" | "-") product)*
//! product    := unary (("*" | "/") unary | unary)*     (juxtaposition)
//! unary      := ("+" | "-") unary | power
//! power      := postfix (("^" | "**") unary)?          (right associative)
//! postfix    := atom "!"*                              (n! = Γ(n+1))
//! atom       := number | name | name "(" expression ("," expression)* ")"
//!             | "(" expression ")"
//! ```
//!
//! Positions and lengths count characters as Swift does for every source
//! the language accepts: Swift counts grapheme clusters (`Character`), and
//! this counts Unicode scalar values (Rust's `char`) but for a CR LF pair,
//! one `Character` in Swift and one position here (`characters`). Other
//! clusters of more than one scalar -- a letter with a combining mark, an
//! emoji sequence -- still count as several here and one in Swift; the
//! language refuses them either way, at a different place
//! (`crates/FINDINGS.md`).

use core::fmt;

use crate::complex::Complex;
use crate::format::format_double;
use crate::gamma::Gamma;
use crate::language::Language;
use crate::tape::Tape;

#[derive(Clone, PartialEq, Debug)]
pub struct ExpressionError {
    pub message: String,
    /// Character offset into the source, and how much of it is at fault, so a
    /// field can underline the mistake. `None` position for empty input.
    pub position: Option<usize>,
    pub length: usize,
}

impl ExpressionError {
    pub fn new(message: impl Into<String>, position: Option<usize>, length: usize) -> ExpressionError {
        ExpressionError { message: message.into(), position, length: length.max(1) }
    }
}

impl fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.position {
            None => write!(f, "{}", self.message),
            Some(p) => write!(f, "{} (at character {})", self.message, p + 1),
        }
    }
}

impl std::error::Error for ExpressionError {}

// MARK: tokens

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TokenKind {
    Number,
    Name,
    Op,
    End,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub position: usize,
}

impl Token {
    /// How many characters the token spans, at least one.
    pub fn length(&self) -> usize {
        self.text.chars().count().max(1)
    }

    /// The token's own characters.
    pub fn count(&self) -> usize {
        self.text.chars().count()
    }

    pub fn is_op(&self, texts: &[&str]) -> bool {
        self.kind == TokenKind::Op && texts.contains(&self.text.as_str())
    }
}

const OPERATORS: [&str; 10] = ["**", "+", "-", "*", "/", "^", "!", "(", ")", ","];

/// Swift's `Character.isLetter`, as near as a `char` has it: Rust's
/// alphabetic property, which is the letters and a few marks and numerals
/// besides (`crates/FINDINGS.md`).
fn is_name_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_name_char(c: char) -> bool {
    c.is_alphabetic() || c.is_numeric() || c == '_'
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// The source as the positions count it: each Unicode scalar value its own
/// slice, but a CR LF pair one, as Swift's `Array(text)` makes it one
/// `Character`.
pub(crate) fn characters(text: &str) -> Vec<&str> {
    let mut out = Vec::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        let width = if rest.starts_with("\r\n") { 2 } else { c.len_utf8() };
        out.push(&rest[..width]);
        rest = &rest[width..];
    }
    out
}

pub(crate) fn tokenize(text: &str) -> Result<Vec<Token>, ExpressionError> {
    // One char per position; a CR LF pair is its CR, which is whitespace like
    // the pair. No token holds whitespace, so a token's text is its chars.
    let chars: Vec<char> = characters(text).iter().map(|u| u.chars().next().expect("a character")).collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    let n = chars.len();
    while i < n {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if is_digit(c) || (c == '.' && i + 1 < n && is_digit(chars[i + 1])) {
            let start = i;
            while i < n && is_digit(chars[i]) {
                i += 1;
            }
            if i < n && chars[i] == '.' {
                i += 1;
                while i < n && is_digit(chars[i]) {
                    i += 1;
                }
            }
            // An `e` is an exponent only when digits follow it; otherwise
            // it is Euler's number and `2e` is a product.
            if i < n && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < n && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < n && is_digit(chars[j]) {
                    i = j;
                    while i < n && is_digit(chars[i]) {
                        i += 1;
                    }
                }
            }
            tokens.push(Token { kind: TokenKind::Number, text: chars[start..i].iter().collect(), position: start });
            continue;
        }
        if is_name_start(c) {
            let start = i;
            while i < n && is_name_char(chars[i]) {
                i += 1;
            }
            tokens.push(Token { kind: TokenKind::Name, text: chars[start..i].iter().collect(), position: start });
            continue;
        }
        let Some(op) = OPERATORS.iter().find(|op| has_prefix(&chars, i, op)) else {
            return Err(ExpressionError::new(format!("'{c}' is not something this language uses"), Some(i), 1));
        };
        tokens.push(Token { kind: TokenKind::Op, text: op.to_string(), position: i });
        i += op.chars().count();
    }
    tokens.push(Token { kind: TokenKind::End, text: String::new(), position: n });
    Ok(tokens)
}

fn has_prefix(chars: &[char], at: usize, prefix: &str) -> bool {
    let p: Vec<char> = prefix.chars().collect();
    at + p.len() <= chars.len() && p.iter().enumerate().all(|(k, c)| chars[at + k] == *c)
}

// MARK: the tree

/// Precedence levels, for deciding where `format` must put parentheses.
const SUM: u8 = 1;
const PRODUCT: u8 = 2;
const UNARY: u8 = 3;
const POWER: u8 = 4;
const POSTFIX: u8 = 5;

#[derive(Clone, PartialEq, Debug)]
pub enum Node {
    Number(f64),
    Variable,
    Constant(String, Complex),
    Call(String, Vec<Node>),
    Binary(String, Box<Node>, Box<Node>),
    Negate(Box<Node>),
    Factorial(Box<Node>),
}

/// The value of a node as a function of z, as `Node::compiled` returns it.
pub type CompiledFn = Box<dyn Fn(Complex) -> Complex + Send + Sync>;

fn parenthesize(text: String, context: u8, mine: u8) -> String {
    if mine < context { format!("({text})") } else { text }
}

impl Node {
    /// Canonical source, the inverse of the parser.
    pub fn format(&self) -> String {
        self.format_in(SUM)
    }

    fn format_in(&self, context: u8) -> String {
        match self {
            Node::Number(v) => format_number_plain(*v),
            Node::Variable => "z".to_string(),
            Node::Constant(name, _) => name.clone(),
            Node::Call(name, args) => {
                format!("{name}({})", args.iter().map(Node::format).collect::<Vec<_>>().join(", "))
            }
            Node::Binary(op, l, r) => {
                if op == "^" {
                    let text = format!("{}^{}", l.format_in(POSTFIX), r.format_in(UNARY));
                    return parenthesize(text, context, POWER);
                }
                let mine = if op == "+" || op == "-" { SUM } else { PRODUCT };
                let text = format!("{}{op}{}", l.format_in(mine), r.format_in(mine + 1));
                parenthesize(text, context, mine)
            }
            Node::Negate(n) => parenthesize(format!("-{}", n.format_in(UNARY)), context, UNARY),
            Node::Factorial(n) => format!("{}!", n.format_in(POSTFIX)),
        }
    }

    /// The value at z. Total: a pole is inf and a branch point nan, never
    /// an error, because a landscape is a picture of where those are.
    ///
    /// Walks the tree: a function is looked up by name and handed its
    /// arguments in an array at every call. Right, and what `compiled` does
    /// once; a landscape evaluates f by the million, and does it through the
    /// tape (`Compiled`).
    pub fn evaluate(&self, z: Complex) -> Complex {
        match self {
            Node::Number(v) => Complex::real(*v),
            Node::Variable => z,
            Node::Constant(_, v) => *v,
            Node::Call(name, args) => {
                let values: Vec<Complex> = args.iter().map(|a| a.evaluate(z)).collect();
                Language::function(name).expect("a call names a function of the table").apply(&values)
            }
            Node::Binary(op, l, r) => {
                let a = l.evaluate(z);
                let b = r.evaluate(z);
                match op.as_str() {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    _ => Complex::pow(a, b),
                }
            }
            Node::Negate(n) => -n.evaluate(z),
            Node::Factorial(n) => Gamma::gamma(n.evaluate(z) + 1.0),
        }
    }

    /// The tree as nested closures, the same values as `evaluate` to the bit:
    /// every name resolved and every operator chosen when the expression is
    /// compiled rather than at each of the million points it is evaluated at,
    /// and a function of one or two arguments called with them rather than
    /// with an array of them.
    pub fn compiled(&self) -> CompiledFn {
        match self {
            Node::Number(v) => {
                let c = Complex::real(*v);
                Box::new(move |_| c)
            }
            Node::Variable => Box::new(|z| z),
            Node::Constant(_, v) => {
                let v = *v;
                Box::new(move |_| v)
            }
            Node::Call(name, args) => {
                let f = Language::function(name).expect("a call names a function of the table");
                let compiled: Vec<CompiledFn> = args.iter().map(Node::compiled).collect();
                if compiled.len() == 1
                    && let Some(unary) = f.unary()
                {
                    let a = compiled.into_iter().next().expect("one argument");
                    return Box::new(move |z| unary(a(z)));
                }
                if compiled.len() == 2
                    && let Some(binary) = f.binary()
                {
                    let mut it = compiled.into_iter();
                    let a = it.next().expect("two arguments");
                    let b = it.next().expect("two arguments");
                    return Box::new(move |z| binary(a(z), b(z)));
                }
                Box::new(move |z| f.apply(&compiled.iter().map(|c| c(z)).collect::<Vec<_>>()))
            }
            Node::Binary(op, l, r) => {
                let a = l.compiled();
                let b = r.compiled();
                match op.as_str() {
                    "+" => Box::new(move |z| a(z) + b(z)),
                    "-" => Box::new(move |z| a(z) - b(z)),
                    "*" => Box::new(move |z| a(z) * b(z)),
                    "/" => Box::new(move |z| a(z) / b(z)),
                    _ => Box::new(move |z| Complex::pow(a(z), b(z))),
                }
            }
            Node::Negate(n) => {
                let a = n.compiled();
                Box::new(move |z| -a(z))
            }
            Node::Factorial(n) => {
                let a = n.compiled();
                Box::new(move |z| Gamma::gamma(a(z) + 1.0))
            }
        }
    }
}

/// A number literal as `Node.format` writes it: an integer without a point
/// below 1e15, otherwise `format_double`. `Swift`'s `String(Int64(v))`.
pub(crate) fn format_number_plain(v: f64) -> String {
    if v == v.round() && v.abs() < 1e15 {
        return (v as i64).to_string();
    }
    format_double(v)
}

// MARK: the parser

struct Parser {
    tokens: Vec<Token>,
    at: usize,
}

impl Parser {
    fn new(text: &str) -> Result<Parser, ExpressionError> {
        Ok(Parser { tokens: tokenize(text)?, at: 0 })
    }

    fn token(&self) -> &Token {
        &self.tokens[self.at]
    }

    fn advance(&mut self) -> Token {
        let t = self.tokens[self.at].clone();
        self.at += 1;
        t
    }

    fn looking_at(&self, texts: &[&str]) -> bool {
        self.token().is_op(texts)
    }

    fn expect(&mut self, text: &str, what: &str) -> Result<Token, ExpressionError> {
        if !self.looking_at(&[text]) {
            let t = self.token();
            return Err(ExpressionError::new(format!("expected '{text}' {what}"), Some(t.position), t.length()));
        }
        Ok(self.advance())
    }

    fn parse(&mut self) -> Result<Node, ExpressionError> {
        if self.token().kind == TokenKind::End {
            return Err(ExpressionError::new("there is no expression here, only empty space", Some(0), 1));
        }
        let node = self.sum()?;
        if self.token().kind != TokenKind::End {
            let t = self.token();
            return Err(ExpressionError::new(
                format!("'{}' has nothing to attach to", t.text),
                Some(t.position),
                t.length(),
            ));
        }
        Ok(node)
    }

    fn sum(&mut self) -> Result<Node, ExpressionError> {
        let mut node = self.product()?;
        while self.looking_at(&["+", "-"]) {
            let op = self.advance().text;
            node = Node::Binary(op, Box::new(node), Box::new(self.product()?));
        }
        Ok(node)
    }

    fn product(&mut self) -> Result<Node, ExpressionError> {
        let mut node = self.unary()?;
        loop {
            if self.looking_at(&["*", "/"]) {
                let op = self.advance().text;
                node = Node::Binary(op, Box::new(node), Box::new(self.unary()?));
            } else if self.starts_an_atom() {
                // Juxtaposition. The right side is a full `unary` so that
                // `2z^2` is `2*(z^2)` rather than `(2*z)^2`.
                node = Node::Binary("*".to_string(), Box::new(node), Box::new(self.unary()?));
            } else {
                return Ok(node);
            }
        }
    }

    fn starts_an_atom(&self) -> bool {
        let t = self.token();
        t.kind == TokenKind::Number || t.kind == TokenKind::Name || (t.kind == TokenKind::Op && t.text == "(")
    }

    fn unary(&mut self) -> Result<Node, ExpressionError> {
        if self.looking_at(&["-"]) {
            self.advance();
            return Ok(Node::Negate(Box::new(self.unary()?)));
        }
        if self.looking_at(&["+"]) {
            self.advance();
            return self.unary();
        }
        self.power()
    }

    fn power(&mut self) -> Result<Node, ExpressionError> {
        let base = self.postfix()?;
        if self.looking_at(&["^", "**"]) {
            self.advance();
            return Ok(Node::Binary("^".to_string(), Box::new(base), Box::new(self.unary()?)));
        }
        Ok(base)
    }

    fn postfix(&mut self) -> Result<Node, ExpressionError> {
        let mut node = self.atom()?;
        while self.looking_at(&["!"]) {
            self.advance();
            node = Node::Factorial(Box::new(node));
        }
        Ok(node)
    }

    fn atom(&mut self) -> Result<Node, ExpressionError> {
        let t = self.token().clone();
        if t.kind == TokenKind::Number {
            self.advance();
            return Ok(Node::Number(t.text.parse::<f64>().unwrap_or(f64::NAN)));
        }
        if t.kind == TokenKind::Name {
            let t = self.advance();
            return self.name(t);
        }
        if self.looking_at(&["("]) {
            self.advance();
            let node = self.sum()?;
            self.expect(")", "to close the group")?;
            return Ok(node);
        }
        if t.kind == TokenKind::End {
            return Err(ExpressionError::new("the expression stops in the middle", Some(t.position), 1));
        }
        Err(ExpressionError::new(format!("'{}' cannot start a term", t.text), Some(t.position), t.length()))
    }

    fn name(&mut self, t: Token) -> Result<Node, ExpressionError> {
        let Some(canonical) = Language::canonical_name(&t.text) else {
            return Err(ExpressionError::new(Language::unknown(&t.text), Some(t.position), t.length()));
        };
        if let Some(f) = Language::function(canonical) {
            if !self.looking_at(&["("]) {
                return Err(ExpressionError::new(
                    format!(
                        "{canonical} is a function; it needs an argument, as in {canonical}({})",
                        if f.arity == 2 { "z, 0.5" } else { "z" }
                    ),
                    Some(t.position),
                    t.length(),
                ));
            }
            self.advance();
            let mut args = vec![self.sum()?];
            while self.looking_at(&[","]) {
                self.advance();
                args.push(self.sum()?);
            }
            let close = self.expect(")", &format!("to close the call to {canonical}"))?;
            if args.len() != f.arity {
                return Err(ExpressionError::new(
                    format!(
                        "{canonical} takes {} argument{}, not {}",
                        f.arity,
                        if f.arity == 1 { "" } else { "s" },
                        args.len()
                    ),
                    Some(t.position),
                    close.position - t.position + 1,
                ));
            }
            return Ok(Node::Call(canonical.to_string(), args));
        }
        if canonical == "z" {
            return Ok(Node::Variable);
        }
        let constant = Language::constant(canonical).expect("a canonical name that is no function is a constant");
        Ok(Node::Constant(canonical.to_string(), constant.value))
    }
}

// MARK: the public three

pub enum Expression {}

impl Expression {
    /// Text to tree, or an `ExpressionError` that says where.
    pub fn parse(text: &str) -> Result<Node, ExpressionError> {
        Parser::new(text)?.parse()
    }

    /// The same expression, spelled one way: aliases resolved, implicit
    /// multiplication explicit, `**` as `^`, spacing gone. Idempotent.
    pub fn canonical(text: &str) -> Result<String, ExpressionError> {
        Ok(Expression::parse(text)?.format())
    }

    /// Text to a function of z.
    pub fn compile(text: &str) -> Result<Compiled, ExpressionError> {
        Ok(Compiled::new(Expression::parse(text)?))
    }
}

/// An expression ready to evaluate by the million, through its `Tape`.
#[derive(Clone)]
pub struct Compiled {
    pub node: Node,
    tape: Tape,
}

impl Compiled {
    pub fn new(node: Node) -> Compiled {
        let tape = Tape::new(&node);
        Compiled { node, tape }
    }

    pub fn expression(&self) -> String {
        self.node.format()
    }

    /// f(z): Swift's `callAsFunction`.
    #[inline]
    pub fn call(&self, z: Complex) -> Complex {
        self.tape.call(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readings_of_the_ambiguous_cases() {
        assert_eq!(Expression::canonical("-z^2").unwrap(), "-z^2");
        assert_eq!(Expression::canonical("2z^2").unwrap(), "2*z^2");
        assert_eq!(Expression::canonical("1/2z").unwrap(), "1/2*z");
        assert_eq!(Expression::canonical("2e").unwrap(), "2*e");
        assert_eq!(Expression::canonical("1e16z").unwrap(), "1e+16*z");
    }

    #[test]
    fn a_hand_built_call_of_another_arity_goes_through_apply() {
        let node = Node::Call("besselj".to_string(), vec![Node::Number(0.0), Node::Variable]);
        let z = Complex::new(1.0, 0.5);
        assert_eq!(node.compiled()(z), node.evaluate(z));
        assert_eq!(Compiled::new(node.clone()).call(z), node.evaluate(z));
    }

    /// Swift reads CR LF as one `Character`: `z\r\n+ @` is refused at
    /// character 4 there, as `Expression.parse` built with swiftc says.
    #[test]
    fn a_cr_lf_pair_is_one_position() {
        let e = Expression::parse("z\r\n+ @").unwrap_err();
        assert_eq!((e.position, e.length), (Some(4), 1));
        assert_eq!(characters("a\r\n\rb\n"), ["a", "\r\n", "\r", "b", "\n"]);
    }

    #[test]
    fn errors_say_where() {
        let e = Expression::parse("sin(z").unwrap_err();
        assert_eq!((e.position, e.length), (Some(5), 1));
        assert_eq!(e.to_string(), "expected ')' to close the call to sin (at character 6)");
    }
}
