# Validation record

Validated November 9, 2026.

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| Qiskit | 2.6.2 |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.8 |
| pytest | 9.2.0 |
| Ruff | 0.17.4 |
| nbformat | 5.11.1 |
| IPython | 9.18.1 |
| pylatexenc | 2.11 |

Completed checks:

- Editable package installation succeeded using the existing dependencies.
- `python -m pytest -q`: **91 passed**.
- The exhaustive Qiskit basis-state test covers 1,360 arithmetic mappings across
  both methods, widths 1--4, and both initial carry-output values.
- Random complex-amplitude checks cover widths 1--5 for both methods.
- Large-input gate-interpreter checks include 257-bit inputs; logical resource
  assertions extend through 1024-bit circuits.
- Exact Clifford+T decomposition counts and a decomposed circuit's quantum
  action were verified separately.
- `python -m ruff check .` and `python -m ruff format --check .` passed.
- All 11 notebook code cells were executed sequentially in an IPython process;
  their assertions and included pytest invocation passed. The execution
  environment disallowed socket-based Jupyter kernels, so the cells were run
  in-process, with captured notebook outputs, rather than through a separate
  kernel connection. The notebook retains a standard Python 3 kernelspec.
- Notebook schema validation passed. Three embedded PNG outputs were generated.
- The circuit diagram and both scaling figures were inspected visually.

The declared package supports Python >=3.10 and Qiskit >=2.0,<3. The versions
above are the versions actually tested; this record does not claim a complete
cross-version or hardware-backend compatibility matrix. No hardware jobs ran.
