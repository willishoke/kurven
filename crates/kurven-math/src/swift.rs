//! Swift's semantics where Rust's differ, for the few operations the port
//! has to spell differently to compute what Swift computes.
//!
//! - `min` and `max`: Swift's generic `min(x, y)` is `y < x ? y : x` and
//!   `max(x, y)` is `y >= x ? y : x`, so a NaN in the first place is kept;
//!   Rust's `f64::min` and `f64::max` drop a NaN wherever it is.
//! - `int`: Swift's `Int(x)` traps on a double that is not finite or does
//!   not fit; Rust's `as` saturates. Where Swift would trap, the port
//!   answers NaN (`crates/FINDINGS.md`), so the callers take `None` as that.

/// Swift's `min(x, y)`: `y < x ? y : x`.
#[inline]
pub fn min(x: f64, y: f64) -> f64 {
    if y < x { y } else { x }
}

/// Swift's `max(x, y)`: `y >= x ? y : x`.
#[inline]
pub fn max(x: f64, y: f64) -> f64 {
    if y >= x { y } else { x }
}

/// Swift's `Int(x)`: the integer part of `x` toward zero, or `None` where
/// Swift traps, on a double that is not finite or not below 2^63 in
/// magnitude (−2^63 itself fits).
#[inline]
pub fn int(x: f64) -> Option<i64> {
    if (-9223372036854775808.0..9223372036854775808.0).contains(&x) { Some(x as i64) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// As `print(max(Double.nan, 16.0), min(Double.nan, 200.0), max(16.0,
    /// Double.nan), min(200.0, Double.nan))` prints, compiled with swiftc on
    /// this machine: `nan nan 16.0 200.0`.
    #[test]
    fn swifts_min_and_max_keep_a_nan_in_the_first_place() {
        assert!(max(f64::NAN, 16.0).is_nan());
        assert!(min(f64::NAN, 200.0).is_nan());
        assert_eq!(max(16.0, f64::NAN), 16.0);
        assert_eq!(min(200.0, f64::NAN), 200.0);
        assert_eq!(max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(min(0.0, -0.0).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn int_is_none_where_swift_traps() {
        assert_eq!(int(-2.7), Some(-2));
        assert_eq!(int(-9223372036854775808.0), Some(i64::MIN));
        assert_eq!(int(9223372036854775808.0), None);
        assert_eq!(int(f64::INFINITY), None);
        assert_eq!(int(f64::NAN), None);
    }
}
