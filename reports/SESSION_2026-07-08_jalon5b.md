# Session report — Milestone 5b: two usage modes, one logic

**Date:** 2026-07-08

## The requirement

QSV must be usable inside any code project WITHOUT the application - two distinct
usage modes, combinable at will.

## Delivered

| Item | Content |
|---|---|
| `qsv/validators.py` | The decision logic extracted from the API into pure functions (`validate_state`, `preparation_qa`) returning typed dataclasses. Zero HTTP, zero FastAPI - importable anywhere. |
| Installable package | `pyproject.toml`, src-layout (`src/qsv/`), `pip install -e ".[api,dev]"`. Package name `quantum-state-validator`, import name `qsv`, version single-sourced from `qsv.__version__` (0.5.0). Extras: `api` (fastapi/uvicorn), `dev` (pytest/httpx/black). |
| `qsv/api.py` | Now a thin HTTP wrapper: endpoints call the library and serialise its dataclasses. Both modes share exactly the same logic - no divergence possible. |
| Rewiring | All imports `src.*` -> `qsv.*` (history preserved via git mv); paths.py anchored at parents[2] with an explicit note that it serves the repo workflow, not library users; 7 notebooks rebranched and re-executed end-to-end (notebook 04 still reproduces the dataset byte-for-byte); CI installs the package itself (`pip install -e`), which now tests the packaging on every push. |
| README | New section "Utilisation - deux modes distincts et combinables" with library and HTTP quickstarts. |

**57 tests green.** One packaging quirk fixed: black inferred a py3.15 target from
`requires-python >= 3.10` (no upper bound) and refused its safety check - pinned
`[tool.black] target-version = ["py310"]`.

## Design statement

Mode 1 (library): `from qsv import validate_state` - for Python pipelines, simulators,
data QA hooks. Mode 2 (service): `uvicorn qsv.api:app` - for any language over HTTP.
Combined: a Python pipeline imports qsv directly while a remote dashboard queries the
same logic over HTTP; the API being a client of the library, the two verdicts cannot
diverge by construction.

## Next

5c: the interactive pedagogical interface (the founding vision). Then the final phase
(full English pass + documentation notebook with the error retrospective).
Progress: ~98%.
