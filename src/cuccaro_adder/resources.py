"""Measured logical resources and optional hardware-independent decomposition."""

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate

from .circuits import _integer, _validate, build_adder


@dataclass(frozen=True)
class ResourceCount:
    """Counts for the circuit actually inspected, in the specified gate basis.

    Each gate has unit weight in depth, including CCX in the logical basis.
    This is not a duration, hardware routing cost, or physical-qubit estimate.
    """

    num_qubits: int
    num_clbits: int
    depth: int
    gate_count: int
    one_qubit_gates: int
    two_qubit_gates: int
    three_plus_qubit_gates: int
    operations: dict[str, int]
    basis: tuple[str, ...] | None

    @property
    def toffoli_count(self) -> int:
        """Number of explicit CCX gates (zero after full decomposition)."""
        return self.operations.get("ccx", 0)

    @property
    def t_count(self) -> int:
        """Number of explicit T and T-dagger gates in the inspected circuit."""
        return self.operations.get("t", 0) + self.operations.get("tdg", 0)

    def to_dict(self) -> dict:
        """Return serializable data suitable for tables or JSON export."""
        return asdict(self) | {"toffoli_count": self.toffoli_count, "t_count": self.t_count}


def count_resources(
    circuit: QuantumCircuit,
    *,
    basis_gates: Sequence[str] | None = None,
    optimization_level: int = 0,
) -> ResourceCount:
    """Inspect a unitary circuit, optionally transpiling to a gate basis.

    Custom composite gates are expanded before counting. Barriers are ignored.
    Circuits with measurement/reset/control flow are rejected so gate_count
    remains a count of quantum gates. Optional transpilation uses no device
    target or coupling map, and therefore adds no hardware routing estimate.
    The input is never modified. Native adder counts exclude preparation;
    counting addition_circuit instead intentionally includes its X gates.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a QuantumCircuit")
    level = _integer(optimization_level, "optimization_level")
    if level > 3:
        raise ValueError("optimization_level must be between 0 and 3")
    expanded = circuit.copy()
    # Flatten ordinary custom subcircuits recursively without decomposing
    # standard CCX gates, which must remain visible in logical resource counts.
    standard = {
        "x",
        "h",
        "cx",
        "ccx",
        "t",
        "tdg",
        "s",
        "sdg",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "p",
        "u",
        "swap",
    }
    while True:
        custom = [
            inst.operation.name
            for inst in expanded.data
            if inst.operation.definition is not None and inst.operation.name not in standard
        ]
        if not custom:
            break
        expanded = expanded.decompose(gates_to_decompose=custom)
    if any(
        inst.operation.name != "barrier" and (inst.clbits or not isinstance(inst.operation, Gate))
        for inst in expanded.data
    ):
        raise ValueError("resource counting requires a unitary circuit without classical control")
    basis = None
    if basis_gates is not None:
        if isinstance(basis_gates, str) or not basis_gates:
            raise ValueError("basis_gates must be a nonempty sequence of gate names")
        basis = tuple(basis_gates)
        expanded = transpile(
            expanded, basis_gates=list(basis), optimization_level=level, seed_transpiler=42
        )
    elif level != 0:
        raise ValueError("optimization_level requires basis_gates")
    instructions = [inst for inst in expanded.data if inst.operation.name != "barrier"]
    operations: dict[str, int] = {}
    for inst in instructions:
        name = inst.operation.name
        operations[name] = operations.get(name, 0) + 1
    return ResourceCount(
        num_qubits=expanded.num_qubits,
        num_clbits=expanded.num_clbits,
        depth=expanded.depth(filter_function=lambda inst: inst.operation.name != "barrier"),
        gate_count=len(instructions),
        one_qubit_gates=sum(len(inst.qubits) == 1 for inst in instructions),
        two_qubit_gates=sum(len(inst.qubits) == 2 for inst in instructions),
        three_plus_qubit_gates=sum(len(inst.qubits) >= 3 for inst in instructions),
        operations=operations,
        basis=basis,
    )


def expected_gate_counts(num_bits: int, *, method: str = "optimized") -> dict[str, int]:
    """Closed-form logical counts; zero-count gate types are omitted."""
    n = _validate(num_bits, method)
    if method == "maj_uma":
        return {"cx": 4 * n + 1, "ccx": 2 * n}
    if n == 1:
        return {"cx": 1, "ccx": 1}
    result = {"cx": 5 * n - 3, "ccx": 2 * n - 1}
    if n > 2:
        result["x"] = 2 * n - 4
    return result


def adder_resources(num_bits: int, *, method: str = "optimized", **kwargs) -> ResourceCount:
    """Build and count the bare adder; kwargs are passed to count_resources."""
    return count_resources(build_adder(num_bits, method=method), **kwargs)


def resource_scaling(widths: Iterable[int], *, method: str = "optimized", **kwargs) -> list[dict]:
    """Return resource rows in input order without exponential simulation."""
    return [
        {"num_bits": n, "method": method, **adder_resources(n, method=method, **kwargs).to_dict()}
        for n in widths
    ]
