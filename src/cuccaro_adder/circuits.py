"""Gate-level constructions, classical input preparation, and output decoding."""

from dataclasses import dataclass
from numbers import Integral

from qiskit import QuantumCircuit, QuantumRegister

METHODS = ("optimized", "maj_uma")


def _integer(value: int, name: str, minimum: int = 0) -> int:
    """Accept integer types, including NumPy integers, but reject bool and floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer (not bool)")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _validate(n: int, method: str) -> int:
    """Centralize width and construction validation for the public functions."""
    n = _integer(n, "num_bits", 1)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return n


def _maj(qc, a, b, carry):
    """Write MAJ(a,b,carry) into a, temporarily modifying b and carry.

    After the two CNOTs, the Toffoli changes a to
    a XOR ((b XOR a) AND (carry XOR a)), which is the majority bit.
    """
    qc.cx(a, b)
    qc.cx(a, carry)
    qc.ccx(carry, b, a)


def _uma(qc, a, b, carry):
    """Undo a MAJ step while depositing the sum in b (the 2-CNOT UMA)."""
    qc.ccx(carry, b, a)
    qc.cx(a, carry)
    qc.cx(carry, b)


def _basic(qc, a, b, work, cout, n):
    """Section 2: ripple the carry upward, copy it, then uncompute downward."""
    for i in range(n):
        _maj(qc, a[i], b[i], work if i == 0 else a[i - 1])
    qc.cx(a[n - 1], cout)
    for i in reversed(range(n)):
        _uma(qc, a[i], b[i], work if i == 0 else a[i - 1])


def _optimized(qc, a, b, work, cout, n):
    """Section 3's scheduled circuit, with explicit small-width boundary cases.

    Figure 5 is written for n>=4. The n=2 and n=3 branches below remove
    overlapping boundary steps. They retain the paper's exact gate counts.
    """
    if n == 1:
        # Half adder: preserve the AND (carry) before overwriting b with XOR.
        qc.ccx(a[0], b[0], cout)
        qc.cx(a[0], b[0])
        return

    # All these gates use disjoint pairs and can run in one time slice.
    for i in range(1, n):
        qc.cx(a[i], b[i])
    qc.cx(a[1], work)

    if n == 2:
        # The most significant carry is written straight into cout.
        qc.ccx(a[0], b[0], work)
        qc.cx(a[1], cout)
        qc.ccx(work, b[1], cout)
        qc.cx(work, b[1])
        qc.ccx(a[0], b[0], work)
        qc.cx(a[1], work)
    elif n == 3:
        # Only one internal MAJ/UMA pair remains between the two endpoints.
        qc.ccx(a[0], b[0], work)
        qc.cx(a[2], a[1])
        qc.ccx(work, b[1], a[1])
        qc.cx(a[2], cout)
        qc.ccx(a[1], b[2], cout)
        qc.x(b[1])
        qc.cx(work, b[1])
        qc.cx(a[1], b[2])
        qc.ccx(work, b[1], a[1])
        qc.ccx(a[0], b[0], work)
        qc.cx(a[2], a[1])
        qc.x(b[1])
        qc.cx(a[1], work)
    else:
        # Forward carry propagation. Each pair of adjacent source lines below
        # follows a time slice in Figure 5; disjoint gates can run in parallel.
        qc.ccx(a[0], b[0], work)
        qc.cx(a[2], a[1])
        qc.ccx(work, b[1], a[1])
        qc.cx(a[3], a[2])
        for i in range(2, n - 2):
            qc.ccx(a[i - 1], b[i], a[i])
            qc.cx(a[i + 2], a[i + 1])
        qc.ccx(a[n - 3], b[n - 2], a[n - 2])
        qc.cx(a[n - 1], cout)

        # Avoid computing the top carry into a and subsequently erasing it:
        # write it directly into the output, saving one Toffoli overall.
        qc.ccx(a[n - 2], b[n - 1], cout)
        for i in range(1, n - 1):
            qc.x(b[i])

        # Begin sum formation. Negations implement the parallel-friendly UMA.
        qc.cx(work, b[1])
        for i in range(2, n):
            qc.cx(a[i - 1], b[i])

        # Reverse propagation restores every a bit and erases intermediate carries.
        qc.ccx(a[n - 3], b[n - 2], a[n - 2])
        for i in range(n - 3, 1, -1):
            qc.ccx(a[i - 1], b[i], a[i])
            qc.cx(a[i + 2], a[i + 1])
            qc.x(b[i + 1])
        qc.ccx(work, b[1], a[1])
        qc.cx(a[3], a[2])
        qc.x(b[2])
        qc.ccx(a[0], b[0], work)
        qc.cx(a[2], a[1])
        qc.x(b[1])
        qc.cx(a[1], work)

    # Finish all sum bits, including the low bit that needed no incoming carry.
    for i in range(n):
        qc.cx(a[i], b[i])


def build_adder(num_bits: int, *, method: str = "optimized") -> QuantumCircuit:
    """Build a reusable, measurement-free unsigned n-bit adder.

    Qubit order: a[0:n], b[0:n], work[0], cout[0], with index 0 the LSB.
    Requires work=0. For arbitrary output target z, its exact mapping is

        |a,b,0,z> -> |a,(a+b) mod 2**n,0,z XOR floor((a+b)/2**n)>.

    Initialize cout=0 to read a full (n+1)-bit sum from cout and b.
    There is no incoming carry input; work is a clean workspace, not carry-in.
    Inputs may be superpositions and may be entangled with other registers.
    """
    n = _validate(num_bits, method)
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    work = QuantumRegister(1, "work")
    cout = QuantumRegister(1, "cout")
    qc = QuantumCircuit(a, b, work, cout, name=f"cuccaro_{method}_{n}")
    if method == "optimized":
        _optimized(qc, a, b, work[0], cout[0], n)
    else:
        _basic(qc, a, b, work[0], cout[0], n)
    return qc


def addition_circuit(
    a: int, b: int, *, num_bits: int | None = None, method: str = "optimized"
) -> QuantumCircuit:
    """Prepare nonnegative integers with X gates, then apply the reusable adder.

    The default width is max(1, a.bit_length(), b.bit_length()). Explicit
    num_bits may zero-pad either input but may not truncate it. Preparation
    gates are included in this circuit's resource count. No simulator is run.
    """
    a = _integer(a, "a")
    b = _integer(b, "b")
    n = max(1, a.bit_length(), b.bit_length()) if num_bits is None else num_bits
    n = _validate(n, method)
    if a.bit_length() > n or b.bit_length() > n:
        raise ValueError("num_bits is too small for the supplied inputs")
    adder = build_adder(n, method=method)
    qc = QuantumCircuit(*adder.qregs, name=f"add_{a}_{b}")
    for i in range(n):
        if (a >> i) & 1:
            qc.x(i)
        if (b >> i) & 1:
            qc.x(n + i)
    qc.compose(adder, inplace=True)
    return qc


@dataclass(frozen=True)
class AdditionOutput:
    """Decoded standard-layout output; total assumes cout initially zero."""

    a: int
    sum_low: int
    work: int
    carry_out: int
    num_bits: int

    @property
    def total(self) -> int:
        """Combine the high carry bit and n low sum bits."""
        return self.sum_low + (self.carry_out << self.num_bits)


def decode_output(index: int, num_bits: int) -> AdditionOutput:
    """Decode a Qiskit basis-state integer in the adder's standard qubit order.

    A measurement string from measure_all() can be passed as int(bits, 2)
    after removing spaces. This decoder assumes the original qubit order;
    hardware layouts or custom classical-bit mappings must be undone first.
    """
    n = _integer(num_bits, "num_bits", 1)
    index = _integer(index, "index")
    if index.bit_length() > 2 * n + 2:
        raise ValueError("index does not fit this circuit's qubit count")
    mask = (1 << n) - 1
    return AdditionOutput(
        a=index & mask,
        sum_low=(index >> n) & mask,
        work=(index >> (2 * n)) & 1,
        carry_out=(index >> (2 * n + 1)) & 1,
        num_bits=n,
    )
