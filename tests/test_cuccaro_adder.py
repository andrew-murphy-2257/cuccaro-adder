"""Arithmetic, reversible quantum behavior, input validation, and resource tests.

Run from the project root with: python -m pytest -q
Small circuits use Qiskit's exact Statevector simulator. Large basis inputs
use an independent X/CX/CCX interpreter, avoiding exponential memory use.
"""

import random

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from cuccaro_adder import (
    adder_resources,
    addition_circuit,
    build_adder,
    count_resources,
    decode_output,
    expected_gate_counts,
    resource_scaling,
)

METHODS = ["optimized", "maj_uma"]


def run_basis_gates(circuit, index):
    """Evaluate actual circuit gates on one basis index, not Python addition.

    X always flips its target. CX flips if its control is 1. CCX flips only
    if both controls are 1. This interpreter cannot test quantum phases;
    the separate complex-superposition tests below cover those explicitly.
    """
    for inst in circuit.data:
        assert inst.operation.name in {"x", "cx", "ccx"}
        positions = [circuit.find_bit(qubit).index for qubit in inst.qubits]
        # An empty control list for X makes all(...) True, as required.
        if all((index >> control) & 1 for control in positions[:-1]):
            index ^= 1 << positions[-1]
    return index


# Every input and both output-target values at widths 1--4 are simulated.
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_exhaustive_qiskit_basis_mapping(n, method):
    """Compare Qiskit evolution with an independently encoded arithmetic oracle."""
    circuit = build_adder(n, method=method)
    for a in range(1 << n):
        for b in range(1 << n):
            for z in (0, 1):
                source = a | (b << n) | (z << (2 * n + 1))
                # Python addition is used only for the expected answer.
                total = a + b
                target = a | ((total % (1 << n)) << n)
                target |= (z ^ (total >> n)) << (2 * n + 1)
                output = Statevector.from_int(source, 1 << circuit.num_qubits).evolve(circuit)
                # Unit amplitude at target implies no leakage, a preserved,
                # correct sum/carry, and work restored to zero.
                assert output.data[target] == pytest.approx(1.0, abs=1e-12)
                assert np.linalg.norm(output.data) == pytest.approx(1.0)


# Random complex amplitudes test phase preservation as well as probabilities.
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_complex_superposition_and_coherent_output_target(n, method):
    """Check a complete coherent arithmetic permutation on the clean-work subspace."""
    rng = np.random.default_rng(1200 + n)
    coefficients = rng.normal(size=2 << (2 * n)) + 1j * rng.normal(size=2 << (2 * n))
    coefficients /= np.linalg.norm(coefficients)
    circuit = build_adder(n, method=method)
    initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
    expected = np.zeros_like(initial)
    for z in (0, 1):
        for a in range(1 << n):
            for b in range(1 << n):
                coefficient = coefficients[a | (b << n) | (z << (2 * n))]
                initial[a | (b << n) | (z << (2 * n + 1))] = coefficient
                total = a + b
                target = a | ((total % (1 << n)) << n)
                target |= (z ^ (total >> n)) << (2 * n + 1)
                expected[target] = coefficient
    actual = Statevector(initial).evolve(circuit).data
    # Comparing complex amplitudes catches relative-phase errors hidden by sampling.
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


# A reversible gate circuit must have a working inverse, even on dirty workspace states.
@pytest.mark.parametrize("method", METHODS)
def test_inverse_on_full_hilbert_space(method):
    """Apply the adder and its inverse to a random full-register state."""
    circuit = build_adder(3, method=method)
    rng = np.random.default_rng(61)
    initial = rng.normal(size=256) + 1j * rng.normal(size=256)
    initial /= np.linalg.norm(initial)
    result = Statevector(initial).evolve(circuit).evolve(circuit.inverse())
    np.testing.assert_allclose(result.data, initial, atol=1e-12, rtol=0)


# These widths cross typical machine-integer boundaries and exercise long carry chains.
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [5, 6, 7, 16, 65, 127, 257])
def test_large_basis_inputs_and_carry_chains(n, method):
    """Check zero, overflow, maximum inputs, asymmetric bits, and seeded random pairs."""
    mask = (1 << n) - 1
    rng = random.Random(777 + n)
    pairs = [
        (0, 0),
        (mask, 0),
        (0, mask),
        (mask, 1),
        (1, mask),
        (mask, mask),
        (1 << (n - 1), 1),
        (mask // 3, 2 * (mask // 3)),
    ]
    pairs.extend((rng.getrandbits(n), rng.getrandbits(n)) for _ in range(25))
    circuit = build_adder(n, method=method)
    for a, b in pairs:
        for z in (0, 1):
            source = a | (b << n) | (z << (2 * n + 1))
            target = a | (((a + b) & mask) << n)
            target |= (z ^ ((a + b) >> n)) << (2 * n + 1)
            assert run_basis_gates(circuit, source) == target


# The wrapper must infer unequal widths correctly, and explicit widths must zero-pad.
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "a,b,n", [(0, 0, None), (0, 1, None), (1, 1, None), (1, 31, None), (13, 2, None), (3, 5, 6)]
)
def test_input_preparation_and_decoding(a, b, n, method):
    """Simulate prepared inputs and check every decoded output field."""
    width = n if n is not None else max(1, a.bit_length(), b.bit_length())
    circuit = addition_circuit(a, b, num_bits=n, method=method)
    state = Statevector.from_instruction(circuit)
    index = int(np.argmax(state.probabilities()))
    result = decode_output(index, width)
    assert state.probabilities()[index] == pytest.approx(1.0)
    assert result.a == a
    assert result.sum_low == (a + b) % (1 << width)
    assert result.carry_out == (a + b) >> width
    assert result.work == 0
    assert result.total == a + b


# All n=2,3 boundary branches and larger widths must follow the published formulas.
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 64, 257, 1024])
def test_logical_counts_and_depth(n, method):
    """Measure the constructed circuit and compare with independent closed forms."""
    resource = adder_resources(n, method=method)
    if method == "optimized":
        expected = {"cx": 1, "ccx": 1} if n == 1 else {"cx": 5 * n - 3, "ccx": 2 * n - 1}
        if n > 2:
            expected["x"] = 2 * n - 4
        assert resource.depth == (2 if n == 1 else 2 * n + 4)
    else:
        expected = {"cx": 4 * n + 1, "ccx": 2 * n}
        # Qiskit schedules disjoint CNOTs in parallel even in this basic chain.
        assert resource.depth == 5 * n + 2
    assert resource.operations == expected
    assert expected_gate_counts(n, method=method) == expected
    assert resource.num_qubits == 2 * n + 2
    assert resource.num_clbits == 0
    assert resource.gate_count == sum(expected.values())
    assert resource.three_plus_qubit_gates == expected["ccx"]
    assert resource.two_qubit_gates == expected["cx"]
    assert resource.one_qubit_gates == expected.get("x", 0)


# A standard exact CCX decomposition uses 6 CX, 7 T/Tdg, and 2 H gates.
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_clifford_t_resource_counts(n, method):
    """Distinguish logical Toffoli resources from elementary decomposition resources."""
    logical = adder_resources(n, method=method)
    basis = ["x", "h", "t", "tdg", "cx"]
    actual = adder_resources(n, method=method, basis_gates=basis)
    assert set(actual.operations) <= set(basis)
    assert actual.toffoli_count == 0
    assert actual.t_count == 7 * logical.toffoli_count
    assert actual.operations["cx"] == logical.operations["cx"] + 6 * logical.toffoli_count
    assert actual.operations["h"] == 2 * logical.toffoli_count
    assert actual.basis == tuple(basis)


def test_decomposed_circuit_has_same_quantum_action():
    """Verify a transpiled elementary-gate circuit against an exact logical output."""
    circuit = addition_circuit(3, 2, num_bits=2)
    elementary = transpile(circuit, basis_gates=["x", "h", "t", "tdg", "cx"], optimization_level=0)
    np.testing.assert_allclose(
        Statevector.from_instruction(elementary).data,
        Statevector.from_instruction(circuit).data,
        atol=1e-12,
    )


def test_resources_preparation_composites_barriers_and_no_mutation():
    """Count the same adder in flat/composite forms and isolate input preparation."""
    bare = build_adder(3)
    original = bare.copy()
    wrapped = QuantumCircuit(bare.num_qubits)
    wrapped.append(bare.to_gate(), wrapped.qubits)
    nested = QuantumCircuit(bare.num_qubits)
    nested.append(wrapped.to_gate(), nested.qubits)
    assert count_resources(nested).operations == count_resources(bare).operations
    bare.barrier()
    assert count_resources(bare).depth == count_resources(original).depth
    assert count_resources(bare).operations == count_resources(original).operations
    prepared = count_resources(addition_circuit(3, 5, num_bits=3))
    assert prepared.gate_count == count_resources(original).gate_count + 4
    count_resources(original, basis_gates=["x", "h", "t", "tdg", "cx"])
    assert original == build_adder(3)


def test_scaling_rows_and_serialization():
    """Preserve width order, support generators/empty input, and export plain data."""
    rows = resource_scaling(n for n in [4, 1, 2])
    assert [row["num_bits"] for row in rows] == [4, 1, 2]
    assert rows[0]["toffoli_count"] == 7
    assert rows[0]["operations"]["cx"] == 17
    assert resource_scaling([]) == []


# Bad widths should fail clearly instead of silently truncating or allocating wrong registers.
@pytest.mark.parametrize(
    "bad,exception",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
        ("3", TypeError),
        (None, TypeError),
    ],
)
def test_invalid_widths(bad, exception):
    """Exercise each public resource/construction entry point's width validation."""
    for function in [build_adder, adder_resources, expected_gate_counts]:
        with pytest.raises(exception):
            function(bad)


# Integer inputs are unsigned; float/bool coercion is deliberately forbidden.
@pytest.mark.parametrize(
    "a,b,exception",
    [
        (-1, 0, ValueError),
        (0, -2, ValueError),
        (1.0, 0, TypeError),
        (True, 0, TypeError),
        (0, "2", TypeError),
    ],
)
def test_invalid_inputs(a, b, exception):
    """Reject malformed operands before any circuit is constructed."""
    with pytest.raises(exception):
        addition_circuit(a, b)


def test_too_narrow_and_invalid_method():
    """Check truncation protection and reject unrecognized algorithm choices."""
    with pytest.raises(ValueError, match="too small"):
        addition_circuit(8, 1, num_bits=3)
    with pytest.raises(ValueError, match="too small"):
        addition_circuit(0, 16, num_bits=4)
    for method in ["unknown", None]:
        with pytest.raises(ValueError, match="method"):
            build_adder(3, method=method)


def test_invalid_decoder_and_resource_requests():
    """Ensure invalid indices and ambiguous/nonunitary resource requests are rejected."""
    for index in [-1, 1 << 8]:
        with pytest.raises(ValueError):
            decode_output(index, 3)
    with pytest.raises(TypeError):
        decode_output(1.0, 3)
    with pytest.raises(ValueError):
        decode_output(0, 0)
    with pytest.raises(TypeError):
        count_resources("not a circuit")
    for kwargs in [
        {"basis_gates": []},
        {"basis_gates": "cx"},
        {"optimization_level": 4},
        {"optimization_level": -1},
        {"optimization_level": 1},
    ]:
        with pytest.raises(ValueError):
            count_resources(build_adder(2), **kwargs)
    measured = build_adder(2)
    measured.measure_all()
    with pytest.raises(ValueError, match="unitary"):
        count_resources(measured)
    reset = QuantumCircuit(1)
    reset.reset(0)
    with pytest.raises(ValueError, match="unitary"):
        count_resources(reset)


def test_numpy_integer_inputs():
    """Accept common NumPy integer scalars while retaining the same arithmetic API."""
    circuit = addition_circuit(np.int64(2), np.int64(3), num_bits=np.int64(3))
    result = run_basis_gates(circuit, 0)
    assert decode_output(result, 3).total == 5


if __name__ == "__main__":
    # Also allow running this file directly as a test program after installation.
    raise SystemExit(pytest.main([__file__, "-q"]))
