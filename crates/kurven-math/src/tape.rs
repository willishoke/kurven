//! An expression as a list of instructions. The port of `Tape.swift`.

use crate::complex::Complex;
use crate::expression::Node;
use crate::gamma::Gamma;
use crate::language::Language;

/// An expression as a list of instructions, each writing one register from
/// the registers before it: the tree in vertical edge-order, the last
/// register the value.
///
/// The same operations as `Node::compiled`'s closures, in the same order, so
/// the same values to the bit. What differs is what an evaluation touches. In
/// Swift a closure tree calls through every node's context, a reference-
/// counted object every thread shares; with the lines of sight of a
/// landscape traversed on every core, most of a core's time went to retaining
/// and releasing them rather than to f. The tape is read through a slice
/// nothing counts, into registers on the stack.
#[derive(Clone, Debug)]
pub struct Tape {
    ops: Vec<Op>,
}

#[derive(Clone, Debug)]
enum Op {
    Constant(Complex),
    Variable,
    Add(usize, usize),
    Subtract(usize, usize),
    Multiply(usize, usize),
    Divide(usize, usize),
    Power(usize, usize),
    Negate(usize),
    Factorial(usize),
    /// A function of the language. Its body captures nothing, so calling it
    /// counts no references.
    Unary(fn(Complex) -> Complex, usize),
    Binary(fn(Complex, Complex) -> Complex, usize, usize),
    /// A call whose arity the table has no direct body for, which the parser
    /// never makes; a tree built by hand can.
    Apply(&'static str, Vec<usize>),
}

/// Registers that fit on the stack; a longer tape takes a vector.
const STACK_REGISTERS: usize = 64;

impl Tape {
    pub fn new(node: &Node) -> Tape {
        let mut list = Vec::new();
        emit(node, &mut list);
        Tape { ops: list }
    }

    pub fn call(&self, z: Complex) -> Complex {
        let n = self.ops.len();
        if n <= STACK_REGISTERS {
            let mut r = [Complex::ZERO; STACK_REGISTERS];
            self.run(z, &mut r[..n])
        } else {
            let mut r = vec![Complex::ZERO; n];
            self.run(z, &mut r)
        }
    }

    fn run(&self, z: Complex, r: &mut [Complex]) -> Complex {
        for (k, op) in self.ops.iter().enumerate() {
            let v = match *op {
                Op::Constant(c) => c,
                Op::Variable => z,
                Op::Add(a, b) => r[a] + r[b],
                Op::Subtract(a, b) => r[a] - r[b],
                Op::Multiply(a, b) => r[a] * r[b],
                Op::Divide(a, b) => r[a] / r[b],
                Op::Power(a, b) => Complex::pow(r[a], r[b]),
                Op::Negate(a) => -r[a],
                Op::Factorial(a) => Gamma::gamma(r[a] + 1.0),
                Op::Unary(f, a) => f(r[a]),
                Op::Binary(f, a, b) => f(r[a], r[b]),
                Op::Apply(name, ref args) => {
                    let values: Vec<Complex> = args.iter().map(|&a| r[a]).collect();
                    Language::function(name).expect("a tape's call names a function of the table").apply(&values)
                }
            };
            r[k] = v;
        }
        r[r.len() - 1]
    }
}

fn emit(n: &Node, list: &mut Vec<Op>) -> usize {
    let op = match n {
        Node::Number(v) => Op::Constant(Complex::real(*v)),
        Node::Variable => Op::Variable,
        Node::Constant(_, v) => Op::Constant(*v),
        Node::Call(name, args) => {
            let f = Language::function(name).expect("a call names a function of the table");
            let registers: Vec<usize> = args.iter().map(|a| emit(a, list)).collect();
            match (registers.len(), f.unary(), f.binary()) {
                (1, Some(unary), _) => Op::Unary(unary, registers[0]),
                (2, _, Some(binary)) => Op::Binary(binary, registers[0], registers[1]),
                _ => Op::Apply(f.name, registers),
            }
        }
        Node::Binary(name, l, r) => {
            let a = emit(l, list);
            let b = emit(r, list);
            match name.as_str() {
                "+" => Op::Add(a, b),
                "-" => Op::Subtract(a, b),
                "*" => Op::Multiply(a, b),
                "/" => Op::Divide(a, b),
                _ => Op::Power(a, b),
            }
        }
        Node::Negate(a) => Op::Negate(emit(a, list)),
        Node::Factorial(a) => Op::Factorial(emit(a, list)),
    };
    list.push(op);
    list.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::Expression;

    /// The tape, the closures and the tree give the same values to the bit.
    #[test]
    fn the_tape_is_the_tree_to_the_bit() {
        let texts = ["gamma(z)^2 - besselj(0.5, z)/z + cn(z, 0.64)!", "-z^2+2z-1/(z^2+1)", "zeta(z) + ψ(z)"];
        for text in texts {
            let node = Expression::parse(text).unwrap();
            let tape = Tape::new(&node);
            let closure = node.compiled();
            for k in 0..50 {
                let z = Complex::new(-3.0 + 0.13 * f64::from(k), 2.0 - 0.07 * f64::from(k));
                let (a, b, c) = (node.evaluate(z), closure(z), tape.call(z));
                assert!(a.re.to_bits() == c.re.to_bits() || (a.re.is_nan() && c.re.is_nan()), "{text} at {z}");
                assert!(a.im.to_bits() == c.im.to_bits() || (a.im.is_nan() && c.im.is_nan()), "{text} at {z}");
                assert!(b.re.to_bits() == c.re.to_bits() || (b.re.is_nan() && c.re.is_nan()), "{text} at {z}");
            }
        }
    }

    #[test]
    fn a_long_tape_spills_to_the_heap() {
        let text = (0..80).map(|k| format!("{k}z")).collect::<Vec<_>>().join("+");
        let node = Expression::parse(&text).unwrap();
        let z = Complex::new(0.5, -0.25);
        assert_eq!(Tape::new(&node).call(z), node.evaluate(z));
    }
}
