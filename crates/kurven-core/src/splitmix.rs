//! SplitMix64 (Steele, Lea and Flood 2014): the generator the suite's
//! property tests draw their cases from (`SplitMix` in
//! `Sources/kurven-test/main.swift`), and its finalizer, which Turing's
//! pattern hashes a lattice point with (`Turing.sign`). One copy, so the
//! tests, the golden sets and the engine mix bits the same way.

/// SplitMix64's finalizer: a bijection of the 64-bit integers that spreads
/// every input bit over the output.
pub fn mix(z: u64) -> u64 {
    let mut z = z;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic generator. The property tests want the same hundred cases
/// every run, so a failure is reproducible rather than a story about last
/// Tuesday. Swift's `SplitMix`, to the bit, so the cases are Swift's.
#[derive(Debug, Clone)]
pub struct SplitMix {
    state: u64,
}

impl SplitMix {
    pub fn new(seed: u64) -> SplitMix {
        SplitMix { state: seed }
    }

    pub fn next_bits(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.state)
    }

    /// A double in `[lo, hi)`, from the top 53 bits.
    pub fn next(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next_bits() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64's published first outputs from seed 0.
    #[test]
    fn the_published_sequence() {
        let mut g = SplitMix::new(0);
        assert_eq!(g.next_bits(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(g.next_bits(), 0x6E78_9E6A_A1B9_65F4);
    }
}
