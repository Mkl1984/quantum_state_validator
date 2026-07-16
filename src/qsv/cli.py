"""
qsv command-line interface - the CI guardrail.

Validate state vectors stored in files, with pytest-like exit codes so a
single line turns any pipeline into a gated one:

    qsv validate states.npy                     # exact-mode check
    qsv validate states.npy --n-shots 500       # finite-shot statistics
    qsv validate dataset.csv --json             # dataset schema (c{i}_real/imag)

Exit codes: 0 = all states valid, 1 = at least one invalid, 2 = usage error.

Supported inputs:
- .npy : complex (or real) array, shape (d,) for one state or (n, d) batch
- .csv : the project dataset schema (c{i}_real / c{i}_imag columns)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from qsv import __version__
from qsv.validators import validate_state

__all__ = ["main"]


def _load_states(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix == ".npy":
        arr = np.asarray(np.load(path)).astype(complex)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"Expected shape (d,) or (n, d), got {arr.shape}")
        return arr
    if path.suffix == ".csv":
        import pandas as pd

        from qsv.features import extract_amplitudes

        return extract_amplitudes(pd.read_csv(path))
    raise ValueError(f"Unsupported file type '{path.suffix}' (use .npy or .csv)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="qsv", description="Quantum State Validator command line"
    )
    parser.add_argument("--version", action="version", version=f"qsv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate state vectors from a file")
    v.add_argument("file", type=Path, help=".npy (complex array) or .csv (dataset)")
    v.add_argument("--n-shots", type=int, default=None, help="measurement budget N")
    v.add_argument(
        "--margin", type=float, default=0.05, help="validity band half-width"
    )
    v.add_argument("--json", action="store_true", help="machine-readable output")
    v.add_argument("--quiet", action="store_true", help="summary line only")

    args = parser.parse_args(argv)

    try:
        states = _load_states(args.file)
    except (OSError, ValueError, TypeError) as exc:
        print(f"qsv: error: {exc}", file=sys.stderr)
        return 2

    results = [
        validate_state(s.real, s.imag, n_shots=args.n_shots, margin=args.margin)
        for s in states
    ]
    n_invalid = sum(1 for r in results if not r.valid)

    if args.json:
        print(
            json.dumps(
                {
                    "file": str(args.file),
                    "n_states": len(results),
                    "n_invalid": n_invalid,
                    "all_valid": n_invalid == 0,
                    "results": [r.to_dict() for r in results],
                }
            )
        )
    else:
        if not args.quiet:
            for i, r in enumerate(results):
                mark = "ok " if r.valid else "FAIL"
                print(
                    f"  [{mark}] state {i}: norm^2 = {r.norm_squared:.6f}"
                    + ("" if r.valid else f"  ({r.explanation})")
                )
        status = "all valid" if n_invalid == 0 else f"{n_invalid} invalid"
        print(
            f"qsv validate: {len(results)} state(s), {status} "
            f"[mode={'noisy N=' + str(args.n_shots) if args.n_shots else 'exact'}]"
        )

    return 0 if n_invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
