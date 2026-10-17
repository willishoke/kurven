import Foundation

extension Expression {
    /// An expression as a list of instructions, each writing one register
    /// from the registers before it: the tree in post-order, the last
    /// register the value.
    ///
    /// The same operations as `Node.compiled()`'s closures, in the same
    /// order, so the same values to the bit. What differs is what an
    /// evaluation touches. A closure tree calls through every node's context,
    /// and a context is a reference-counted object every thread shares: with
    /// the sight lines of a landscape marched on every core, most of a core's
    /// time went to retaining and releasing them -- the cores passing the
    /// same few cache lines back and forth -- rather than to f. The tape is
    /// read through a pointer nothing counts, into registers on the stack.
    final class Tape: @unchecked Sendable {
        enum Op {
            case constant(Complex)
            case variable
            case add(Int, Int)
            case subtract(Int, Int)
            case multiply(Int, Int)
            case divide(Int, Int)
            case power(Int, Int)
            case negate(Int)
            case factorial(Int)
            /// A function of the language. Its body captures nothing, so
            /// calling it counts no references.
            case unary(@Sendable (Complex) -> Complex, Int)
            case binary(@Sendable (Complex, Complex) -> Complex, Int, Int)
            /// A call whose arity the table has no direct body for, which the
            /// parser never makes; a tree built by hand can.
            case apply(@Sendable ([Complex]) -> Complex, [Int])
        }

        /// Never written after `init`, and freed by `deinit` alone.
        private let ops: UnsafeMutableBufferPointer<Op>

        init(_ node: Node) {
            var list: [Op] = []
            @discardableResult
            func emit(_ n: Node) -> Int {
                let op: Op
                switch n {
                case .number(let v): op = .constant(Complex(v))
                case .variable: op = .variable
                case .constant(_, let v): op = .constant(v)
                case .call(let name, let args):
                    let f = Language.function(name)!
                    let registers = args.map { emit($0) }
                    if registers.count == 1, let unary = f.unary {
                        op = .unary(unary, registers[0])
                    } else if registers.count == 2, let binary = f.binary {
                        op = .binary(binary, registers[0], registers[1])
                    } else {
                        op = .apply(f.apply, registers)
                    }
                case .binary(let name, let l, let r):
                    let a = emit(l), b = emit(r)
                    switch name {
                    case "+": op = .add(a, b)
                    case "-": op = .subtract(a, b)
                    case "*": op = .multiply(a, b)
                    case "/": op = .divide(a, b)
                    default: op = .power(a, b)
                    }
                case .negate(let a): op = .negate(emit(a))
                case .factorial(let a): op = .factorial(emit(a))
                }
                list.append(op)
                return list.count - 1
            }
            emit(node)
            ops = .allocate(capacity: list.count)
            _ = ops.initialize(from: list)
        }

        deinit {
            ops.deinitialize()
            ops.deallocate()
        }

        func callAsFunction(_ z: Complex) -> Complex {
            let ops = self.ops
            let n = ops.count
            return withUnsafeTemporaryAllocation(of: Complex.self, capacity: n) { r in
                for k in 0..<n {
                    let v: Complex
                    switch ops[k] {
                    case .constant(let c): v = c
                    case .variable: v = z
                    case .add(let a, let b): v = r[a] + r[b]
                    case .subtract(let a, let b): v = r[a] - r[b]
                    case .multiply(let a, let b): v = r[a] * r[b]
                    case .divide(let a, let b): v = r[a] / r[b]
                    case .power(let a, let b): v = Complex.pow(r[a], r[b])
                    case .negate(let a): v = -r[a]
                    case .factorial(let a): v = Gamma.gamma(r[a] + 1)
                    case .unary(let f, let a): v = f(r[a])
                    case .binary(let f, let a, let b): v = f(r[a], r[b])
                    case .apply(let f, let args): v = f(args.map { r[$0] })
                    }
                    r.initializeElement(at: k, to: v)
                }
                return r[n - 1]
            }
        }
    }
}
