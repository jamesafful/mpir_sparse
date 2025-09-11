# mpir-sparse (starter)

Adaptive **mixed-precision iterative refinement** for sparse linear systems, with an **adaptive scheduler** and optional **ILU/AMG preconditioning**. Includes reproducible benchmarks on the **SuiteSparse Matrix Collection**.

## Quickstart (GitHub Codespaces)

```bash
# In Codespaces terminal:
pip install -e ".[dev]"
pytest -q
python benchmarks/run_suitesparse.py --limit 3 --plot
jupyter notebook notebooks/01_quick_demo.ipynb
```

## Layout

- `src/mpir_sparse/`
  - `ir.py`: iterative refinement core
  - `schedulers.py`: adaptive policies
  - `preconditioners.py`: ILU + AMG hooks
  - `utils.py`: helpers (norms, operators, diagnostics)
- `benchmarks/run_suitesparse.py`: download and run experiments (uses `ssgetpy`), plotting
- `tests/test_ir.py`: minimal correctness tests
- `notebooks/`: quick demo + SuiteSparse walkthrough

## License

MIT
