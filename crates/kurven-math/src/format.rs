//! Numbers as text, the way the Swift reference writes them.
//!
//! The expression language's canonical spelling of a number is Swift's
//! `Double.description` (`Expression.formatDouble`), which is Python's
//! `repr(float)` on every number `tests/fixtures/expr/language.json` holds.
//! Rust's own `Display` and `Debug` lay digits out differently (`1e16`, never
//! `1e+16`), so the layout is written out here; the digits themselves are
//! Rust's shortest round-trip ones, as Swift's are.

/// Swift's `Double.description`: the shortest decimal that reads back as the
/// same double, written in positional notation when 1e-4 <= |v| <= 2^53 and
/// otherwise with an exponent of at least two digits and its sign
/// (`1e+16`, `1.5e-05`, `5e-324`). Measured against `swiftc` on this
/// machine: 2^53 is written `9007199254740992.0`, the next double above it
/// `9.007199254740994e+15`; `0.0001` positionally, the double below it
/// `9.999999999999999e-05`. A NaN is `nan` whatever its sign, as Swift writes
/// it; zero is `0.0` or `-0.0`.
pub fn description(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".to_string() } else { "inf".to_string() };
    }
    if v == 0.0 {
        return if crate::real::sign_minus(v) { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let sign = if v < 0.0 { "-" } else { "" };
    let (digits, exponent) = shortest_digits(v.abs());
    let positional = exponent >= -4 && v.abs() <= 9007199254740992.0;
    if !positional {
        let mut out = String::from(sign);
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let e = exponent.unsigned_abs();
        if e < 10 {
            out.push('0');
        }
        out.push_str(&e.to_string());
        return out;
    }
    let mut out = String::from(sign);
    if exponent < 0 {
        out.push_str("0.");
        for _ in 0..(-exponent - 1) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let whole = exponent as usize + 1;
        if digits.len() <= whole {
            out.push_str(&digits);
            for _ in digits.len()..whole {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..whole]);
            out.push('.');
            out.push_str(&digits[whole..]);
        }
    }
    out
}

/// The shortest round-trip significant digits of a positive finite `v`, and
/// the decimal exponent of the first: 0.00125 is ("125", -3).
fn shortest_digits(v: f64) -> (String, i32) {
    let text = format!("{v:e}");
    let (mantissa, exponent) = text.split_once('e').expect("{:e} writes an exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    (digits, exponent.parse().expect("{:e} writes an integer exponent"))
}

/// Python's `repr(float)`: the shortest round-tripping decimal, with an
/// exponent written as `e-05` rather than `e-5`. (`Expression.formatDouble`,
/// which rewrites Swift's `description`; Swift writes that form already, and
/// the rewriting is ported as it is.)
pub fn format_double(v: f64) -> String {
    let s = description(v);
    let Some(e) = s.find(['e', 'E']) else { return s };
    let mut mantissa = s[..e].to_string();
    let mut exponent = s[e + 1..].to_string();
    let mut sign = "+";
    if let Some(rest) = exponent.strip_prefix('-') {
        sign = "-";
        exponent = rest.to_string();
    } else if let Some(rest) = exponent.strip_prefix('+') {
        exponent = rest.to_string();
    }
    if exponent.chars().count() < 2 {
        exponent = format!("0{exponent}");
    }
    if mantissa.ends_with(".0") {
        mantissa.truncate(mantissa.len() - 2);
    }
    format!("{mantissa}e{sign}{exponent}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each against what `print("\(v)")` writes, compiled with swiftc on
    /// this machine.
    #[test]
    fn swifts_description() {
        let cases: [(f64, &str); 27] = [
            (1e15, "1000000000000000.0"),
            (1.5e15, "1500000000000000.0"),
            (9007199254740992.0, "9007199254740992.0"),
            (9.5e15, "9.5e+15"),
            (1e16, "1e+16"),
            (1.25e16, "1.25e+16"),
            (123456789012345.6, "123456789012345.6"),
            (0.0001, "0.0001"),
            (0.00012, "0.00012"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (5e-324, "5e-324"),
            (1e300, "1e+300"),
            (f64::MAX, "1.7976931348623157e+308"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
            (0.1, "0.1"),
            (100.5, "100.5"),
            (1e22, "1e+22"),
            (12345678901234567890.0, "1.2345678901234567e+19"),
            (-0.0, "-0.0"),
            (2.5e-7, "2.5e-07"),
            (9007199254740994.0, "9.007199254740994e+15"),
            (9007199254740990.0, "9007199254740990.0"),
            (0.00009999999999999999, "9.999999999999999e-05"),
            (-9.5e15, "-9.5e+15"),
        ];
        for (v, want) in cases {
            assert_eq!(description(v), want, "{v:e}");
        }
        assert_eq!(description(9007199254740991.0), "9007199254740991.0");
        assert_eq!(description(-1e-5), "-1e-05");
        assert_eq!(description(8e15), "8000000000000000.0");
    }

    #[test]
    fn format_double_is_reprs() {
        assert_eq!(format_double(1e-5), "1e-05");
        assert_eq!(format_double(1e16), "1e+16");
        assert_eq!(format_double(0.001), "0.001");
        assert_eq!(format_double(2.5), "2.5");
    }
}
