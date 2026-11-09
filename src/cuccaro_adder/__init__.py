"""Cuccaro addition constructed explicitly from X, CX, and CCX gates.

No Qiskit arithmetic library or synthesis routine is used to build the adder.
The default implements Cuccaro et al., quant-ph/0410184, Section 3 / Figure 5.
"""

from .circuits import AdditionOutput, addition_circuit, build_adder, decode_output
from .resources import (
    ResourceCount,
    adder_resources,
    count_resources,
    expected_gate_counts,
    resource_scaling,
)

__all__ = [
    "AdditionOutput",
    "ResourceCount",
    "adder_resources",
    "addition_circuit",
    "build_adder",
    "count_resources",
    "decode_output",
    "expected_gate_counts",
    "resource_scaling",
]
