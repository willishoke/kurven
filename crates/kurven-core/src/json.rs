//! A JSON value, its reader, and its canonical writer: the port of
//! `Sources/KurvenCore/JSON.swift`.
//!
//! `Codable` is deliberately not used for the manifest in Swift, and serde
//! is not used here, for the same reason. The contract test is "decode a
//! manifest, re-encode it, compare bytes with the file", and the writer must
//! put a number down exactly as Swift's `Double.description` does (5.0 keeps
//! its point; past 2^53 it turns to an exponent), which no stock serializer
//! does. Owning the writer costs a page and buys an exact comparison, plus a
//! decoder whose errors name the key that was wrong.
//!
//! As in Swift, `Int` and `Double` are different values: the files write
//! `int` for counts and `float` for measurements, and the canonical writer
//! must give each back as it was. A number written with a fraction or an
//! exponent is a `Double`; one without is an `Int`, as `JSONSerialization`
//! boxes an integral literal as a long long. A double's text is read by
//! Rust's `str::parse`, which is correctly rounded, so a number round trips
//! to the bit.
//!
//! Objects keep their keys sorted (`BTreeMap`), which is the order the
//! canonical form writes; the reader refuses a key given twice rather than
//! keep one silently.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Double(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

/// An object's members: Swift's `[String: JSONValue]`.
pub type JsonObject = BTreeMap<String, JsonValue>;

/// Why a document could not be read, or a value in it was not what was
/// asked for: Swift's `JSONError`, whose descriptions these are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonError {
    NotAnObject(String),
    MissingKey {
        key: String,
        within: String,
    },
    WrongType {
        key: String,
        expected: String,
        found: String,
    },
    /// The text is not JSON: where the reader stopped, in bytes, and why.
    BadSyntax {
        offset: usize,
        message: String,
    },
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonError::NotAnObject(w) => write!(f, "json: {w} is not an object"),
            JsonError::MissingKey { key, within } => write!(f, "json: {within} has no key '{key}'"),
            JsonError::WrongType { key, expected, found } => write!(f, "json: '{key}' in {found} should be {expected}"),
            JsonError::BadSyntax { offset, message } => write!(f, "json: {message} at byte {offset}"),
        }
    }
}

impl std::error::Error for JsonError {}

impl JsonError {
    pub fn wrong_type(key: &str, expected: &str, found: &str) -> JsonError {
        JsonError::WrongType { key: key.to_string(), expected: expected.to_string(), found: found.to_string() }
    }
}

impl JsonValue {
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?.get(key)
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// An `Int`, or a `Double` with an integral value, as Swift's typed
    /// access reads a count.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Int(i) => Some(*i),
            JsonValue::Double(d) if d.fract() == 0.0 && d.abs() < 9.007_199_254_740_992e15 => Some(*d as i64),
            _ => None,
        }
    }

    /// A `Double`, or an `Int` as a double, as Swift's typed access reads a
    /// measurement.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Double(d) => Some(*d),
            JsonValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
}

/// The value `text` holds: one JSON document, with nothing after it but
/// whitespace.
pub fn parse(text: &str) -> Result<JsonValue, JsonError> {
    let mut reader = Reader { bytes: text.as_bytes(), at: 0 };
    reader.whitespace();
    let value = reader.value(0)?;
    reader.whitespace();
    if reader.at != reader.bytes.len() {
        return Err(reader.error("text after the document"));
    }
    Ok(value)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

/// Deep enough for any manifest, shallow enough that a hostile file cannot
/// overflow the stack.
const MAX_DEPTH: usize = 128;

impl Reader<'_> {
    fn error(&self, message: &str) -> JsonError {
        JsonError::BadSyntax { offset: self.at, message: message.to_string() }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn literal(&mut self, word: &str, value: JsonValue) -> Result<JsonValue, JsonError> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.error("unknown literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nested too deeply"));
        }
        match self.peek() {
            None => Err(self.error("the document stops in the middle")),
            Some(b'n') => self.literal("null", JsonValue::Null),
            Some(b't') => self.literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.literal("false", JsonValue::Bool(false)),
            Some(b'"') => Ok(JsonValue::String(self.string()?)),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.whitespace();
                if self.peek() == Some(b']') {
                    self.at += 1;
                    return Ok(JsonValue::Array(items));
                }
                loop {
                    self.whitespace();
                    items.push(self.value(depth + 1)?);
                    self.whitespace();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(JsonValue::Array(items));
                        }
                        _ => return Err(self.error("expected ',' or ']'")),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut object = BTreeMap::new();
                self.whitespace();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                    return Ok(JsonValue::Object(object));
                }
                loop {
                    self.whitespace();
                    if self.peek() != Some(b'"') {
                        return Err(self.error("expected a key"));
                    }
                    let key_at = self.at;
                    let key = self.string()?;
                    self.whitespace();
                    if self.peek() != Some(b':') {
                        return Err(self.error("expected ':'"));
                    }
                    self.at += 1;
                    self.whitespace();
                    let value = self.value(depth + 1)?;
                    if object.insert(key, value).is_some() {
                        return Err(JsonError::BadSyntax { offset: key_at, message: "a key given twice".into() });
                    }
                    self.whitespace();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(JsonValue::Object(object));
                        }
                        _ => return Err(self.error("expected ',' or '}'")),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
        }
    }

    fn number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.at;
        let mut integral = true;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.at += 1;
                }
            }
            _ => return Err(self.error("a number without digits")),
        }
        if self.peek() == Some(b'.') {
            integral = false;
            self.at += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("a fraction without digits"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            integral = false;
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("an exponent without digits"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
        }
        // ASCII by construction.
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| self.error("not UTF-8"))?;
        if integral && let Ok(i) = text.parse::<i64>() {
            return Ok(JsonValue::Int(i));
        }
        text.parse::<f64>()
            .map(JsonValue::Double)
            .map_err(|_| JsonError::BadSyntax { offset: start, message: format!("not a number: {text}") })
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let digits = self.bytes.get(self.at..self.at + 4).ok_or_else(|| self.error("a short \\u escape"))?;
        let text = std::str::from_utf8(digits).map_err(|_| self.error("a bad \\u escape"))?;
        let unit = u32::from_str_radix(text, 16).map_err(|_| self.error("a bad \\u escape"))?;
        self.at += 4;
        Ok(unit)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.at += 1; // the opening quote
        let mut out = String::new();
        loop {
            let run = self.at;
            while matches!(self.peek(), Some(c) if c != b'"' && c != b'\\' && c >= 0x20) {
                self.at += 1;
            }
            out.push_str(std::str::from_utf8(&self.bytes[run..self.at]).map_err(|_| self.error("not UTF-8"))?);
            match self.peek() {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escaped = self.peek().ok_or_else(|| self.error("the string stops in an escape"))?;
                    self.at += 1;
                    match escaped {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            let scalar = if (0xD800..0xDC00).contains(&unit) {
                                if !self.bytes[self.at..].starts_with(b"\\u") {
                                    return Err(self.error("a lone high surrogate"));
                                }
                                self.at += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(self.error("a high surrogate without its low one"));
                                }
                                0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                unit
                            };
                            out.push(char::from_u32(scalar).ok_or_else(|| self.error("a lone low surrogate"))?);
                        }
                        _ => return Err(self.error("an unknown escape")),
                    }
                }
                None => return Err(self.error("the string stops in the middle")),
                Some(_) => return Err(self.error("a control character in a string")),
            }
        }
    }
}

// MARK: - the canonical writer

impl JsonValue {
    /// Canonical form: sorted keys, no insignificant whitespace, and numbers
    /// formatted as Swift's `Double.description` formats them (Python's
    /// `repr` but past 2^53, where Swift turns to an exponent and Python
    /// waits for 1e16). Two documents are equal iff their canonical forms are
    /// byte-equal.
    pub fn canonical(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn write_canonical(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Int(i) => out.push_str(&i.to_string()),
            JsonValue::Double(d) => out.push_str(&number(*d)),
            JsonValue::String(s) => out.push_str(&quote(s, false)),
            JsonValue::Array(a) => {
                out.push('[');
                for (k, v) in a.iter().enumerate() {
                    if k > 0 {
                        out.push(',');
                    }
                    v.write_canonical(out);
                }
                out.push(']');
            }
            JsonValue::Object(o) => {
                out.push('{');
                for (k, (key, v)) in o.iter().enumerate() {
                    if k > 0 {
                        out.push(',');
                    }
                    out.push_str(&quote(key, false));
                    out.push(':');
                    v.write_canonical(out);
                }
                out.push('}');
            }
        }
    }

    /// The indented document Swift's `JSONValue.pretty(indent:ascii:)` writes,
    /// byte for byte: the layout of `json.dumps(value, sort_keys=True,
    /// indent=indent, ensure_ascii=ascii)`, so the fixture indexes read as
    /// Python wrote them, with Swift's strings and numbers (`quote`,
    /// `number`). Those are not Python's everywhere: `\b` and `\f` are
    /// written `\u0008` and `\u000c`, DEL is not escaped under `ascii`, and
    /// a double from 2^53 up to 1e16 takes an exponent. (Swift's defaults
    /// are an indent of 1 and `ascii`.)
    pub fn pretty(&self, indent: usize, ascii: bool) -> String {
        fn render(v: &JsonValue, depth: usize, indent: usize, ascii: bool) -> String {
            let inner = " ".repeat(indent * (depth + 1));
            let outer = " ".repeat(indent * depth);
            match v {
                JsonValue::Array(a) if !a.is_empty() => {
                    let items: Vec<String> =
                        a.iter().map(|x| format!("{inner}{}", render(x, depth + 1, indent, ascii))).collect();
                    format!("[\n{}\n{outer}]", items.join(",\n"))
                }
                JsonValue::Object(o) if !o.is_empty() => {
                    let items: Vec<String> = o
                        .iter()
                        .map(|(k, x)| format!("{inner}{}: {}", quote(k, ascii), render(x, depth + 1, indent, ascii)))
                        .collect();
                    format!("{{\n{}\n{outer}}}", items.join(",\n"))
                }
                JsonValue::Array(_) => "[]".to_string(),
                JsonValue::Object(_) => "{}".to_string(),
                JsonValue::String(s) => quote(s, ascii),
                _ => v.canonical(),
            }
        }
        render(self, 0, indent, ascii)
    }
}

/// A double as Swift's `Double.description` writes it, which is Python's
/// `repr` up to 2^53: the shortest decimal that round-trips, with a
/// trailing `.0` on integral values, in positional notation from 1e-4 up to
/// 2^53 and as `d.ddde±XX` outside it. Refuses the values JSON cannot
/// represent rather than emit an unparseable `inf`.
///
/// The digits are Rust's shortest round-trip digits (`{:e}`), which are
/// Swift's: both give the shortest digit string that reads back to the
/// double, the nearest such where there are several.
pub fn number(d: f64) -> String {
    assert!(d.is_finite(), "JSON cannot represent {d}; clamp before encoding");
    if d == 0.0 {
        return if d.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let sign = if d < 0.0 { "-" } else { "" };
    let scientific = format!("{:e}", d.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("{:e} writes an exponent");
    let exponent: i32 = exponent.parse().expect("{:e} writes an integral exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if d.abs() <= 9_007_199_254_740_992.0 && exponent >= -4 {
        if exponent < 0 {
            let zeros = "0".repeat((-exponent - 1) as usize);
            return format!("{sign}0.{zeros}{digits}");
        }
        let point = exponent as usize + 1;
        if digits.len() <= point {
            let zeros = "0".repeat(point - digits.len());
            return format!("{sign}{digits}{zeros}.0");
        }
        return format!("{sign}{}.{}", &digits[..point], &digits[point..]);
    }
    let exponent_sign = if exponent < 0 { '-' } else { '+' };
    let fraction = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
    format!("{sign}{}{fraction}e{exponent_sign}{:02}", &digits[..1], exponent.unsigned_abs())
}

/// A string as JSON, as Swift's `JSONValue.quote` writes it: quotes,
/// backslash escapes for `"`, `\`, newline, return and tab, `\u00XX` for
/// the other control characters (`\b` and `\f` among them, which Python
/// writes as `\b` and `\f`), and with `ascii` every scalar past 0x7F as
/// its UTF-16 code units, `\uXXXX`. DEL, 0x7F, is written as it is, where
/// Python's `ensure_ascii` escapes it.
pub fn quote(s: &str, ascii: bool) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        if ascii && (c as u32) > 0x7F {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
            continue;
        }
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// MARK: - typed access

/// Whether a number read as a double is a whole number an `Int` holds:
/// 1e30 is whole, and is refused rather than trapped on.
pub fn is_int(d: f64) -> bool {
    d == d.round() && d.abs() < 9.223_372_036_854_776e18 // 2^63
}

impl JsonValue {
    /// The members of an object, `what` naming it in a refusal.
    pub fn object(&self, what: &str) -> Result<&JsonObject, JsonError> {
        self.as_object().ok_or_else(|| JsonError::NotAnObject(what.to_string()))
    }

    /// An array of numbers, `key` in `what` naming it in a refusal.
    pub fn doubles(&self, key: &str, what: &str) -> Result<Vec<f64>, JsonError> {
        let refuse = || JsonError::wrong_type(key, "an array of numbers", what);
        let a = self.as_array().ok_or_else(refuse)?;
        a.iter()
            .map(|v| match v {
                JsonValue::Double(d) => Ok(*d),
                JsonValue::Int(i) => Ok(*i as f64),
                _ => Err(refuse()),
            })
            .collect()
    }

    /// An array of integers, as `doubles` reads numbers.
    pub fn ints(&self, key: &str, what: &str) -> Result<Vec<i64>, JsonError> {
        let refuse = || JsonError::wrong_type(key, "an array of integers", what);
        let a = self.as_array().ok_or_else(refuse)?;
        a.iter()
            .map(|v| match v {
                JsonValue::Int(i) => Ok(*i),
                JsonValue::Double(d) if is_int(*d) => Ok(*d as i64),
                _ => Err(refuse()),
            })
            .collect()
    }
}

/// Typed reads of an object's members, each naming the key and the object
/// (`what`) when it refuses: Swift's extension on `[String: JSONValue]`.
pub trait Members {
    fn value(&self, key: &str, what: &str) -> Result<&JsonValue, JsonError>;
    fn string(&self, key: &str, what: &str) -> Result<String, JsonError>;
    fn int(&self, key: &str, what: &str) -> Result<i64, JsonError>;
    fn double(&self, key: &str, what: &str) -> Result<f64, JsonError>;
    /// A number, or `None` where the key is absent or holds anything else:
    /// Swift's `optionalDouble`, which a wrong type does not refuse.
    fn optional_double(&self, key: &str) -> Option<f64>;
    /// A boolean, or `fallback` where the key is absent; with no fallback an
    /// absent key is refused.
    fn bool(&self, key: &str, what: &str, fallback: Option<bool>) -> Result<bool, JsonError>;
    fn array(&self, key: &str, what: &str) -> Result<&[JsonValue], JsonError>;
    fn doubles(&self, key: &str, what: &str) -> Result<Vec<f64>, JsonError>;
    fn ints(&self, key: &str, what: &str) -> Result<Vec<i64>, JsonError>;
    fn strings(&self, key: &str, what: &str) -> Result<Vec<String>, JsonError>;
}

impl Members for JsonObject {
    fn value(&self, key: &str, what: &str) -> Result<&JsonValue, JsonError> {
        self.get(key).ok_or_else(|| JsonError::MissingKey { key: key.to_string(), within: what.to_string() })
    }

    fn string(&self, key: &str, what: &str) -> Result<String, JsonError> {
        match self.value(key, what)? {
            JsonValue::String(s) => Ok(s.clone()),
            _ => Err(JsonError::wrong_type(key, "a string", what)),
        }
    }

    fn int(&self, key: &str, what: &str) -> Result<i64, JsonError> {
        match self.value(key, what)? {
            JsonValue::Int(i) => Ok(*i),
            JsonValue::Double(d) if is_int(*d) => Ok(*d as i64),
            _ => Err(JsonError::wrong_type(key, "an integer", what)),
        }
    }

    fn double(&self, key: &str, what: &str) -> Result<f64, JsonError> {
        match self.value(key, what)? {
            JsonValue::Double(d) => Ok(*d),
            JsonValue::Int(i) => Ok(*i as f64),
            _ => Err(JsonError::wrong_type(key, "a number", what)),
        }
    }

    fn optional_double(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(JsonValue::Double(d)) => Some(*d),
            Some(JsonValue::Int(i)) => Some(*i as f64),
            _ => None,
        }
    }

    fn bool(&self, key: &str, what: &str, fallback: Option<bool>) -> Result<bool, JsonError> {
        let Some(v) = self.get(key) else {
            return fallback.ok_or_else(|| JsonError::MissingKey { key: key.to_string(), within: what.to_string() });
        };
        match v {
            JsonValue::Bool(b) => Ok(*b),
            _ => Err(JsonError::wrong_type(key, "a boolean", what)),
        }
    }

    fn array(&self, key: &str, what: &str) -> Result<&[JsonValue], JsonError> {
        match self.value(key, what)? {
            JsonValue::Array(a) => Ok(a),
            _ => Err(JsonError::wrong_type(key, "an array", what)),
        }
    }

    fn doubles(&self, key: &str, what: &str) -> Result<Vec<f64>, JsonError> {
        self.value(key, what)?.doubles(key, what)
    }

    fn ints(&self, key: &str, what: &str) -> Result<Vec<i64>, JsonError> {
        self.value(key, what)?.ints(key, what)
    }

    fn strings(&self, key: &str, what: &str) -> Result<Vec<String>, JsonError> {
        self.array(key, what)?
            .iter()
            .map(|v| match v {
                JsonValue::String(s) => Ok(s.clone()),
                _ => Err(JsonError::wrong_type(key, "an array of strings", what)),
            })
            .collect()
    }
}

/// An object from its members, as Swift writes `.object([...])`.
pub fn object<const N: usize>(members: [(&str, JsonValue); N]) -> JsonValue {
    JsonValue::Object(members.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

pub fn string(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}

pub fn doubles(xs: &[f64]) -> JsonValue {
    JsonValue::Array(xs.iter().map(|&x| JsonValue::Double(x)).collect())
}

pub fn ints(xs: &[i64]) -> JsonValue {
    JsonValue::Array(xs.iter().map(|&x| JsonValue::Int(x)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_and_doubles_stay_apart() {
        assert_eq!(parse("5").unwrap(), JsonValue::Int(5));
        assert_eq!(parse("5.0").unwrap(), JsonValue::Double(5.0));
        assert_eq!(parse("-0").unwrap(), JsonValue::Int(0));
        assert_eq!(parse("1e2").unwrap(), JsonValue::Double(100.0));
        assert_eq!(parse("-2.5E-3").unwrap(), JsonValue::Double(-0.0025));
        // Past i64: a double, as JSONSerialization reads it.
        assert_eq!(parse("9223372036854775808").unwrap(), JsonValue::Double((1u64 << 63) as f64));
    }

    /// Numbers round trip to the bit: the shortest text of a double reads
    /// back as that double.
    #[test]
    fn numbers_read_back_to_the_bit() {
        for x in [0.1, 1.0 / 3.0, 2.0f64.sqrt(), 5e-324, f64::MAX, -core::f64::consts::PI] {
            assert_eq!(parse(&format!("{x:?}")).unwrap(), JsonValue::Double(x));
        }
    }

    #[test]
    fn strings_and_escapes() {
        assert_eq!(parse(r#""a\"b\\c\/\n\t\u00e9\ud83d\ude00""#).unwrap(), JsonValue::String("a\"b\\c/\n\té😀".into()));
        assert_eq!(parse("\"Γ(z)\"").unwrap(), JsonValue::String("Γ(z)".into()));
    }

    #[test]
    fn structures() {
        let v = parse(r#" { "b": [1, 2.5, null, true, false], "a": {} , "c": [] } "#).unwrap();
        assert_eq!(v.as_object().unwrap().keys().collect::<Vec<_>>(), ["a", "b", "c"]);
        let b = v.get("b").unwrap().as_array().unwrap();
        assert_eq!(b[0].as_i64(), Some(1));
        assert_eq!(b[1].as_f64(), Some(2.5));
        assert_eq!(b[2], JsonValue::Null);
        assert_eq!(b[3].as_bool(), Some(true));
    }

    #[test]
    fn refusals() {
        for bad in [
            "",
            "[1,]",
            "{\"a\":1,}",
            "01",
            "1.",
            ".5",
            "1e",
            "nul",
            "\"abc",
            "\"\\x\"",
            "[1 2]",
            "{\"a\":1,\"a\":2}",
            "\"\\ud800\"",
            "\"\t\"",
            "1 2",
            "-",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} was read");
        }
        assert!(parse(&"[".repeat(200)).is_err());
    }

    /// "contract: canonical JSON writes numbers the way Python's json does".
    /// The reason Codable is not used in Swift: JSONEncoder renders 5.0 as
    /// "5", and a manifest that round trips to a different document is not
    /// a contract.
    #[test]
    fn canonical_json_writes_numbers_the_way_pythons_json_does() {
        assert_eq!(JsonValue::Double(5.0).canonical(), "5.0", "5.0 keeps its point");
        assert_eq!(JsonValue::Double(-79.5).canonical(), "-79.5");
        assert_eq!(JsonValue::Double(0.51).canonical(), "0.51", "0.51 is shortest-round-trip");
        assert_eq!(JsonValue::Int(5).canonical(), "5", "an integer stays an integer");
        assert_eq!(
            object([("b", JsonValue::Int(1)), ("a", JsonValue::Int(2))]).canonical(),
            r#"{"a":2,"b":1}"#,
            "keys are sorted, separators tight"
        );
    }

    /// Every case Swift's `Double.description` was asked on this machine
    /// (swift 6, 2026-11-15), with what it wrote.
    #[test]
    fn numbers_are_written_as_swift_writes_them() {
        for (x, want) in [
            (5.0, "5.0"),
            (-79.5, "-79.5"),
            (0.51, "0.51"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (0.00012345, "0.00012345"),
            (9.5e-5, "9.5e-05"),
            (9.99999e-5, "9.99999e-05"),
            (0.00010000000000000002, "0.00010000000000000002"),
            (1.1e-4, "0.00011"),
            (1e-4 * 0.9999999999999999, "9.999999999999999e-05"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (8.5e15, "8500000000000000.0"),
            (9007199254740900.0, "9007199254740900.0"),
            (9007199254740991.0, "9007199254740991.0"),
            (9007199254740992.0, "9007199254740992.0"),
            (9007199254740994.0, "9.007199254740994e+15"),
            (18014398509481984.0, "1.8014398509481984e+16"),
            (1.5e16, "1.5e+16"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (5e-324, "5e-324"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (2.5e-7, "2.5e-07"),
            (1e22, "1e+22"),
            (1e300, "1e+300"),
            (f64::MAX, "1.7976931348623157e+308"),
            (4503599627370496.5, "4503599627370496.0"),
            (123.456, "123.456"),
            (0.001, "0.001"),
        ] {
            assert_eq!(number(x), want, "{x:e}");
            assert_eq!(parse(want).unwrap().as_f64(), Some(x), "{want} reads back");
        }
    }

    #[test]
    fn quoting_and_the_pretty_form() {
        assert_eq!(quote("a\"b\\c\n\u{1}é", false), "\"a\\\"b\\\\c\\n\\u0001é\"");
        assert_eq!(quote("Γ😀", true), "\"\\u0393\\ud83d\\ude00\"");
        // Swift's, not Python's: \b and \f by number, DEL as it is.
        assert_eq!(quote("\u{8}\u{c}\u{7f}", true), "\"\\u0008\\u000c\u{7f}\"");
        let v = object([("b", ints(&[1, 2])), ("a", object([])), ("c", string("x"))]);
        assert_eq!(v.pretty(1, true), "{\n \"a\": {},\n \"b\": [\n  1,\n  2\n ],\n \"c\": \"x\"\n}");
    }

    #[test]
    fn typed_access_names_what_it_refuses() {
        let v = parse(r#"{"n": 3, "x": 2.0, "s": "t", "l": [1, 2.5], "b": true, "w": 1e30}"#).unwrap();
        let o = v.object("thing").unwrap();
        assert_eq!(o.int("n", "thing"), Ok(3));
        assert_eq!(o.int("x", "thing"), Ok(2));
        assert_eq!(o.double("n", "thing"), Ok(3.0));
        assert_eq!(o.doubles("l", "thing"), Ok(vec![1.0, 2.5]));
        assert_eq!(o.bool("b", "thing", None), Ok(true));
        assert_eq!(o.bool("absent", "thing", Some(false)), Ok(false));
        assert_eq!(o.int("w", "thing").unwrap_err().to_string(), "json: 'w' in thing should be an integer");
        assert_eq!(o.string("absent", "thing").unwrap_err().to_string(), "json: thing has no key 'absent'");
        assert_eq!(string("x").object("Thing").unwrap_err().to_string(), "json: Thing is not an object");
    }
}
