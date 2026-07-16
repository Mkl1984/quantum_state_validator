"""
Module: paths.py
Purpose: project-root-anchored paths, independent of the current working
directory.

Why this module exists (audit A1)
---------------------------------
Relative paths like ``"data/processed"`` depend on the working directory
at execution time. Run from ``notebooks/``, they created a duplicate of
the dataset (1.8 MB) and a stray ``notebooks/models/`` folder. This module
anchors every path on the repository root, derived from this file's own
location - which is invariant whatever the cwd.

Usage
-----
>>> from qsv.paths import DATA_PROCESSED, MODELS_DIR
>>> df = pd.read_csv(DATA_PROCESSED / "quantum_states_10000.csv")
"""

from pathlib import Path

#: Repository root: this file lives at <root>/src/qsv/, hence parents[2].
#: NOTE: these paths serve the REPOSITORY workflow (notebooks, dataset
#: regeneration). When qsv is pip-installed inside another project, use your
#: own paths - the library functions never depend on this module.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

#: Main dataset (10,000 states, dim=4, balanced 50/50).
MAIN_DATASET: Path = DATA_PROCESSED / "quantum_states_10000.csv"
