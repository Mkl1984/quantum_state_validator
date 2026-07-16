# Web app acceptance protocol (Emergent build "qsv-learn")

The web app mirrors the qsv library; before any release or design iteration is
accepted, its backend must pass the vectors below, computed with the reference
implementation (`qsv/validators.py`, package v0.5.0). Preview:
https://qsv-learn.preview.emergentagent.com/

## Decision rules (must match exactly)

- Exact mode: valid iff |norm2 - 1| <= 1e-6, norm2 = sum(re^2 + im^2)
- Noisy mode: sigma = 1/(2*sqrt(N)); bias = 2*d*sigma^2;
  statistic = |norm2 - 1 - bias|; valid iff statistic <= margin/2;
  budget_sufficient = (2*sigma <= margin/2), surfaced as a warning when false
- Preparation QA: F = |sum(conj(t_i)*s_i)|^2 / norm2; gain ok iff
  |norm2-1| <= margin; pointing ok iff F >= 0.95; error_type in
  {ok, gain_error, pointing_error, gain_and_pointing_error}

## Test vectors (imag = 0 unless stated; margin = 0.05)

| # | Input | Expected |
|---|---|---|
| T1 | validate [0.6, 0.8, 0, 0], exact | valid=true, norm2=1.0 |
| T2 | validate [0.66, 0.88, 0, 0], exact | valid=false, norm2=1.21 |
| T3 | validate [0.62, 0.8, 0, 0], N=500 | valid=true, norm2=1.0244, sigma=0.022361, bias=0.004, statistic=0.0204, threshold=0.025, **budget_sufficient=false (warning shown despite valid verdict)** |
| T4 | validate [0.6, 0.8, 0, 0], N=10 | **valid=false although the true state is valid** (bias=0.2 dominates; statistic=0.2), budget_sufficient=false, noise-dominated verdict explained |
| T5 | prep-qa state [0.78, 1.04, 0, 0] vs target [0.6, 0.8, 0, 0] | prep_ok=false, error_type=gain_error, norm2=1.69, fidelity=1.0 |
| T6 | prep-qa state [sqrt(.7)*.6, sqrt(.7)*.8, sqrt(.3), 0] vs same target | prep_ok=false, error_type=pointing_error, norm2=1.0, fidelity=0.7 |

## Content checks

- V1: lesson-4 sizing numbers exactly 19.4% / 8.9% / 3.4% / 1.1% / <0.05%
  for N = 25 / 100 / 400 / 1600 / 6400; no invented numbers anywhere
- V2: no emoji in the UI
- V3: all UI text in English

## Design constraints (carry into any redesign)

Dark default, deep purple, sober; KaTeX math; T3/T4 behaviours prominent in the
Validator page; a Bloch sphere may only appear in a clearly labeled
"the qubit case (d=2)" section - never to visualize d=4 playground states.

## Status log

- 2026-07-16: acceptance message queued to the Emergent agent; the account ran
  out of credits immediately after - execution unconfirmed. To resume: top up,
  ask the agent for the T1-T6/V1-V3 report, then (all-PASS only) launch design
  Phase 1. Tag v0.5.0 is blocked until the report is all-PASS.
