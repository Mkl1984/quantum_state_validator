# Session report — Milestone 4c: preparation QA against known targets

**Date:** 2026-07-07 (evening session)
**New project conventions received and applied** (versioned in ROADMAP.md): English
everywhere from now on (existing French content to be translated in the final polish
phase), lean code notebooks, final documentation notebook with a first-person error
retrospective built exclusively from the real, documented project history, no emoji,
no over-engineering. This session's code, notebook and report are in English.

## Delivered

| Item | Content |
|---|---|
| `src/preparation.py` | Preparation-QA generator against Haar-random known targets. Error model: `ok` / `rotated` (unitary pointing error, norm-blind) / `scaled` (gain error, fidelity-blind), all read through shot noise. `fidelity_to_target` is scale-invariant by construction. |
| Notebook 11 (executed, English) | Experiments run before interpretations, as always. |
| Tests | +5 (target normalisation, fidelity bounds/scale-invariance, per-class statistical signatures, reproducibility, validation). **Total: 45 green.** |

## Scientific results

| Binary task (accept/reject) | Accuracy |
|---|---|
| Norm statistic alone | 0.6667 |
| Fidelity alone | 0.6653 |
| **Pair (norm, fidelity), depth-3 tree** | **0.9218** |
| RF on raw amplitudes + target id | 0.9071 |

3-class diagnosis: 0.925 with the pair; **`rotated` diagnosed at 100%** — the notebook 09
isotropy limit (40% scaling/noise confusion, provably irreducible without a reference)
is broken exactly as predicted once the validator knows the intended target. Residual
ok/scaled confusion (~9-14%) is shot noise at the band edge: reducible by increasing N,
not by a better model.

Each single statistic lands at exactly ~2/3 — blind to precisely one of three balanced
classes. The engineered pair beats the 100-tree forest on raw data: fourth appearance
of the representation lesson since notebook 08.

## Milestone 4 status

4a (correlated noise: negative result owned), 4b (calibration drift: hybrid wins),
4c (known targets: isotropy broken) — the milestone's core question is answered.
Remaining as 4d: variable dimensions, N-vs-error sizing curves. Then milestone 5
(API + pedagogical interface) and the final phase (full English pass + documentation
notebook with error retrospective). Progress: ~95%.
