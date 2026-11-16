//! A JSON value and its reader: the reading half of
//! `Sources/KurvenCore/JSON.swift`.
//!
//! As in Swift, `Int` and `Double` are different values: the files write
//! `int` for counts and `float` for measurements, and the canonical writer
//! (Phase 2, the bundle schema) must give each back as it was. A number
//! written with a fraction or an exponent is a `Double`; one without is an
//! `Int`, as `JSONSerialization` boxes an integral literal as a long long.
//! A double's text is read by Rust's `str::parse`, which is correctly
//! rounded, so a number round trips to the bit.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    /// Byte offset into the text where the reader stopped.
    pub offset: usize,
    pub message: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json: {} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for JsonError {}

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
        JsonError { offset: self.at, message: message.to_string() }
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
                        return Err(JsonError { offset: key_at, message: "a key given twice".into() });
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
            .map_err(|_| JsonError { offset: start, message: format!("not a number: {text}") })
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
}
