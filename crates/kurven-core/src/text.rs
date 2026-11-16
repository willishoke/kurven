//! Numbers in prose, as Swift's interpolation and `String(format:)` write
//! them, so a refusal reads the same in the port as in the reference.

use crate::json;

/// A double as Swift's `"\(x)"` writes it: `Double.description`, with
/// `inf`, `-inf` and `nan` for the values JSON cannot hold.
pub fn double(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        json::number(x)
    }
}

/// An array of doubles as Swift's `"\(xs)"` writes it: `[1.0, 2.5]`.
pub fn doubles(xs: &[f64]) -> String {
    format!("[{}]", xs.iter().map(|&x| double(x)).collect::<Vec<_>>().join(", "))
}

/// An array of strings as Swift's `"\(xs)"` writes it: `["a", "b"]`.
pub fn strings(xs: &[String]) -> String {
    format!("[{}]", xs.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(", "))
}

/// An array of integers as Swift's `"\(xs)"` writes it: `[1, 2]`.
pub fn ints(xs: &[i64]) -> String {
    format!("[{}]", xs.iter().map(i64::to_string).collect::<Vec<_>>().join(", "))
}

/// C's `%.3g`, as `String(format: "%.3g", x)` writes it: three significant
/// digits, trailing zeros dropped, an exponent below 1e-4 and from 1e3 up.
pub fn g3(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let scientific = format!("{x:.2e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("{:e} writes an exponent");
    let exponent: i32 = exponent.parse().expect("an integral exponent");
    fn trimmed(s: &str) -> &str {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
    }
    if !(-4..3).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trimmed(mantissa), exponent.unsigned_abs())
    } else {
        let decimals = (2 - exponent) as usize;
        trimmed(&format!("{x:.decimals$}")).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `String(format: "%.3g", x)` writes, as printf writes it.
    #[test]
    fn three_significant_digits_as_printf_writes_them() {
        for (x, want) in [
            (1.0, "1"),
            (0.5, "0.5"),
            (123.0, "123"),
            (1234.0, "1.23e+03"),
            (16_777_216.0, "1.68e+07"),
            (0.0001234, "0.000123"),
            (0.00001234, "1.23e-05"),
            (999.6, "1e+03"),
            (65537.0, "6.55e+04"),
            (-2.5, "-2.5"),
            (1e30, "1e+30"),
        ] {
            assert_eq!(g3(x), want, "{x}");
        }
        assert_eq!(double(f64::INFINITY), "inf");
        assert_eq!(doubles(&[1.0, f64::NAN]), "[1.0, nan]");
        assert_eq!(strings(&["a".to_string(), "b".to_string()]), "[\"a\", \"b\"]");
    }
}
