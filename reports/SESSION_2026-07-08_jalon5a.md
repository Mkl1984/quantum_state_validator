# Session report — Milestone 5a: the prediction API

**Date:** 2026-07-08

## The architecture decision

The API serves the validators that WON the milestone-4 benchmark, not a trained model:
the bias-corrected threshold test (notebook 08) for anonymous validation and the
(norm, fidelity) two-channel monitor (notebook 11) for known-target preparation QA.
Rationale, written into the module docstring: a statistic is auditable, sizeable
against a measurement budget (notebook 12 curve backs the API's budget-sufficiency
warning), and has no training-drift failure mode. Serving the science's actual
conclusion instead of a model for the sake of it IS the deliverable.

## Delivered

| Item | Content |
|---|---|
| `src/api.py` (EN) | FastAPI app: `/health`, `/validate` (exact mode = strict computation, the notebook 07 lesson; noisy mode = bias-corrected statistic vs margin/2 + budget warning), `/preparation-qa` (gain channel + pointing channel, failing channel names the error type). Pedagogical explanation strings in every response. OpenAPI docs at /docs. |
| Tests | 7 TestClient tests (channels, budget warning, validation errors). **Total: 57 green.** |
| Infra | fastapi/uvicorn/httpx in requirements; CI install updated (the API is tested in CI). |

## Bug caught by the tests

`sigma_from_shots` returns a NumPy float64; comparisons then produce `numpy.bool`,
which FastAPI cannot JSON-serialise. One failing test, explicit float()/bool() casts
at the API boundary, documented in a comment. The kind of integration bug unit tests
of the physics could never catch - exactly why the API has its own test file.

## Next

5b: the interactive pedagogical interface (the project's founding vision). Then the
final phase: full English pass + documentation notebook with the first-person error
retrospective. Progress: ~97%.
