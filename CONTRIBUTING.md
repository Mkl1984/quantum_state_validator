# Contributing to Quantum State Validator

Thank you for your interest! This project is a learning environment for
quantum mechanics and honest ML: **scientific rigor and pedagogy outrank
everything else**. A contribution that makes the code faster but less
understandable will be declined; a contribution that documents an
instructive mistake will be celebrated (see notebook 07).

## Setup

```bash
git clone https://github.com/Mkl1984/quantum_state_validator.git
cd quantum_state_validator
python -m venv venv && source venv/bin/activate   # .\venv\Scripts\Activate.ps1 on Windows
pip install -e ".[api,dev]"
pytest tests/ -q          # the whole suite must pass before you start
```

## Project rules

1. **No target leakage.** Any new training feature must be classified as
   scale-invariant or scale-sensitive in `qsv/features.py`, with an
   invariance test if it claims to be invariant. The
   `test_invariant_features_are_scale_invariant` test is non-negotiable.
2. **The class boundary is unique**: |norm^2 - 1| >= `norm_margin` for every
   invalid state (see `data/README.md`). Never bypass it locally.
3. **Libraries do not talk**: `logging`, never `print()` inside `src/`.
4. **Paths go through `qsv/paths.py`** - never cwd-relative paths in
   repository workflows.
5. **Every public function** carries a docstring with the formulas, the
   hypotheses, and the physical meaning of the result.
6. **Every behaviour change** comes with a test that locks it.

## Style and quality

- Formatting: `black src/ tests/` (checked in CI, blocking)
- Tests: `pytest tests/ -q` (blocking in CI, Python 3.10 and 3.12)
- Notebooks: executed end to end before committing, outputs included
- Language: everything in English (code, comments, docs, figures)

## Git workflow

- Branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>` from `main`
- Commits: [Conventional Commits](https://www.conventionalcommits.org/) in
  English - `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`
- One commit = one logical change. The message explains the *why*.
- Pull requests target `main`: CI must be green; describe any scientific
  impact (features, class boundary, noise model)

## Reporting an issue

Open an issue with: Python version, OS, minimal reproduction steps,
expected vs observed behaviour. For a scientific question (formulation,
noise model, features), cite the relevant README section or notebook.

## License

By contributing, you agree that your contribution is released under the
MIT license (see LICENSE).
