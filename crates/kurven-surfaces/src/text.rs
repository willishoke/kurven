//! C's `%g`, as Swift's `String(format:)` writes it, for the messages that
//! carry a number so.

/// `%.<precision>g`: `precision` significant digits, positional where the
/// exponent is from -4 to below the precision, trailing zeros dropped, the
/// exponent signed and of two digits at least. As `traversal.rs`'s `g4`,
/// for any precision.
pub fn g(x: f64, precision: usize) -> String {
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".to_string()
        } else if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let p = precision.max(1);
    let scientific = format!("{x:.*e}", p - 1);
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i64 = exponent.parse().unwrap_or(0);
    let trim = |s: String| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    };
    if exponent >= -4 && exponent < p as i64 {
        let decimals = (p as i64 - 1 - exponent).max(0) as usize;
        trim(format!("{x:.decimals$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa.to_string()), exponent.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::g;

    #[test]
    fn numbers_are_written_as_c_writes_them() {
        assert_eq!(g(0.5, 3), "0.5");
        assert_eq!(g(12.34567, 4), "12.35");
        assert_eq!(g(1.0e-5, 3), "1e-05");
        assert_eq!(g(0.000123456, 2), "0.00012");
        assert_eq!(g(123456.0, 3), "1.23e+05");
        assert_eq!(g(1234.0, 2), "1.2e+03");
    }
}
