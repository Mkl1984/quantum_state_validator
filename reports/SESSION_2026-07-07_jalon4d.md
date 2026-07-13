# Session report — Milestone 4d: scaling and sizing. Milestone 4 closed (v0.4.0)

**Date:** 2026-07-07 (late session)

## Incident worth recording

The working sandbox was recycled mid-session and the local clone (with the first,
unsynced 4d run) was lost. Recovery took minutes and the re-run produced **bit-identical
numbers** - the seed discipline built in Phase 1/2 paid for itself in the most concrete
way possible. Lesson reinforced: sync early, and reproducibility is not a formality.

## Delivered

| Item | Content |
|---|---|
| Notebook 12 (executed, English) | Dimension scaling + measurement-budget sizing curve. |
| Tests | +5 parametrized (invariant-feature bounds at d in {2,8,16}, noise pipeline at d in {2,8}). **Total: 50 green.** |
| Release | Tag v0.4.0, CHANGELOG 0.4.0 section cut. |

## Scientific results

1. **Counter-intuitive and owned**: threshold-test accuracy IMPROVES with dimension
   (0.973 at d=2 to 0.993 at d=16, fixed N=500). Concentration of measure works FOR the
   validator: invalid norm deviations accumulate over 2d components while the shot noise
   on a valid state's norm statistic stays d-independent at first order. Caveat stated:
   holds for this generator's invalid population, not for margin-adversarial states.
2. **Sizing curve** (d=4, margin 0.05): FPR 19% at N=25, 1.1% at N=1600, below test-set
   resolution at N=6400. Engineering statement: FPR <= 1% requires N ~ 2000 shots.
   Margin and budget trade like stability vs integration time in a GNSS clock budget.
   FNR is never the binding constraint.

## Milestone 4 — final picture

Stationary iid noise: threshold test (08). Common-mode noise: threshold test, negative
result for ML owned (09). Anonymous diagnosis: Bayes limit by isotropy (09).
Calibration drift: hybrid physics+ML (10). Known-target QA: engineered pair, isotropy
broken (11). Scaling: d-agnostic, sizing curve (12).

**One sentence: ML earns its place when the problem is non-stationary or diagnostic,
and performs best married to the physics, not opposed to it.**

## Next

Milestone 5 (prediction API + pedagogical interface), then the final phase: full English
pass and the project documentation notebook with the first-person error retrospective
(source material: reports/ and git history only). Progress: ~96%.
