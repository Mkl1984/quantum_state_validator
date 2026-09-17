# Session report - Exact counting model: the last P0, and what it cost the project

**Date:** 2026-09-17
**Commits:** `3aaf4ab`, `65139e6`, `92251bb`
**Tests:** 81 -> 104

## The requirement

Every noise model in notebooks 08-12 adds Gaussian noise to the amplitudes with
sigma = 1 / (2 sqrt(N)). That was declared as a simplification in the README, in the
ROADMAP and in the "hypotheses and limits" chapter of the project report - the single
largest open scientific debt of the repository. The requirement was to replace it with
the model a counting experiment actually realises, and then to measure honestly what
changes.

## Delivered

| Item | Content |
|---|---|
| `qsv/tomography.py` | Exact finite-shot model. `sample_counts` draws k_i ~ Poisson(N \|c_i\|^2); `estimate_norm_squared` returns the unbiased estimator with its plug-in variance; `conditional_probabilities` exposes the scale-invariant part of the data as a first-class object; `validate_counts` decides with the same geometry as `validate_state` but no bias correction, and reports an exact two-sided Poisson p-value. |
| Exact p-value | Computed by direct summation in log space over a 40-sigma window. No scipy dependency added to the package; cross-checked against `scipy.stats.poisson` at 1e-12 agreement or better before being trusted. |
| Notebook 15 | Code-only, executed, one pointer cell. Five experiments: bias/variance of the estimator, the 2d sigma^2 comparison across (d, N), the overlaid sampling distributions, the two-panel non-identifiability figure, and the notebook-12 sizing curve recomputed under both models. |
| Notebook 13 | New section "Notebook 15 - Exact finite-shot counting model"; the limits chapter rewritten; the future-work list updated (two entries closed, one promoted to main open gap); repository standing corrected. |
| `qsv/__init__.py`, README, ROADMAP, CHANGELOG | Public exports, a usage section for raw detector counts, and the closure of the P0 code scope. |
| Cleanup | `src/test_vscode.py` removed - see below. |

## Why Poisson rather than multinomial

The ROADMAP entry said "multinomial". Writing it made the reason to deviate obvious,
and the deviation is the session's main scientific point.

A multinomial over the d outcomes is what you obtain *after conditioning on the total
count*. Conditioning on the total is exactly what destroys the norm: under
k -> k / sum(k) the global scale cancels identically. So a single-multinomial model is
norm-blind by construction, and a model that cannot represent the quantity under test
cannot be the model for this project. The Poisson counting model keeps the total as a
random variable and therefore keeps the norm observable.

The notebook proves this twice rather than asserting it. Conditional frequencies for
states at \|\|psi\|\|^2 = 0.85, 1.00 and 1.20 coincide to three decimals while their
total counts differ by 41% (339017 / 398934 / 478832); the two-panel figure puts the
separated total-count distributions beside the superimposed frequency distributions.

This is the scale-invariance of notebook 07 arriving from the opposite direction. There,
a scale-*sensitive* feature answered the question so completely that it leaked the
label. Here, a scale-*invariant* summary cannot answer it at all. Same geometry, two
failure modes, and the engineering consequence is concrete: a norm check requires a
calibrated exposure. An instrument reporting only relative frequencies has already
thrown the answer away.

## What the experiments found

Run before any interpretation was written, as since notebook 09.

1. **The counting estimator is unbiased.** Largest measured bias over 200000
   repetitions: 4.5e-4, roughly two Monte-Carlo standard errors (resolution 2.2e-4),
   with the twelve measurements scattered around zero and no systematic sign. The
   spread follows sqrt(\|\|psi\|\|^2 / N) to within 0.3% everywhere.

2. **The 2d sigma^2 correction is an artefact of the Gaussian model.** The Gaussian
   bias tracks the theoretical value across every (d, N) tested (+0.08018 against
   0.08000 at d = 16, N = 100). The counting bias stays inside its own Monte-Carlo
   resolution and, the discriminating observation, does not grow with d at all. This
   does not invalidate the bias-corrected test of notebook 08: within its own model the
   correction is exactly right, and `validators.py` keeps it for that reason. It
   relocates it. The correction compensates for squaring additive amplitude noise, and
   a counting detector never squares anything.

3. **Both models agree on the spread.** sd 0.0498 (counting) against 0.0500 (Gaussian)
   at d = 4, N = 400; algebraically both give 1/sqrt(N) at unit norm. The
   sigma = 1/(2 sqrt(N)) choice made in notebook 08 was right for the quantity that
   actually drives the decision.

4. **The sizing curve survives.** Recomputing the notebook-12 experiment under both
   models: the FPR gap never exceeds 0.6 point, the FNR gap never exceeds 1.7 points,
   both largest at the smallest budget where every rate is noise-dominated anyway, and
   the differences are not systematically in one direction. Milestones 1 to 4 stand as
   published.

**This is the fourth negative-style result the project has kept rather than buried**,
after the Random Forest that did not beat the threshold test under correlated noise and
the one that did not learn the drift. A model refinement that did not change the
conclusion is not a wasted session - it is the only thing that converts "declared
simplification" into "measured approximation", and the difference matters when someone
asks whether the numbers mean anything.

`budget_ok` also gains a closed form: one standard deviation must fit inside the
tolerance, 1/sqrt(N) <= margin/2, that is N >= 4/margin^2, which is 1600 at margin 0.05.
The analytical version of the empirical curve of notebook 12 - and the two agree.

## Errors and incidents

**I wrote three figures the outputs did not support.** After drafting the notebook-13
section I re-read it against the notebook's own tables and found: "agree to four
decimals" (they agree to three), "bias never leaves the 1e-4 band" (2.2e-4), "FNR
agreeing within about one point" (1.7 points at the worst budget). All three were
overclaims in the same direction - tighter than the truth. The text and the commit
message were corrected before the work was pushed. Cost: about twenty minutes. The
lesson is narrower than "run experiments first", which I did: **re-reading the
interpretation against the table is a separate step from writing it**, and rounding
drifts optimistic when you already believe the conclusion.

**The `.git` sync lost its Windows settings.** The established workflow - work in a
`/tmp` clone, `git gc`, copy `.git` back - produced a repository reporting all 67
tracked files as modified. Cause: a clone made inside the Linux sandbox writes
`filemode = true`, while the mounted Windows checkout presents every file as 0755, so
every path showed a spurious 100644 -> 100755 mode change. `core.symlinks` and
`core.ignorecase` were lost the same way. Fix: re-apply `core.filemode false`,
`core.symlinks false`, `core.ignorecase true` after every `.git` copy. Cost: about
fifteen minutes of a genuinely alarming `git status`. Added to the workflow permanently.

**`src/test_vscode.py`.** The mode-flag investigation surfaced the one file still
carrying CRLF endings, and it turned out to be an editor-setup scratch file from the
first days of the project, sitting inside the package directory with French comments, a
deliberately unused variable, a deliberately over-long line, and a `test_` prefix on
something that is not a test. It survived the audit, the English translation pass and
the packaging refactor. Removed. Worth recording that a mechanical inconsistency, not a
review, is what found it.

## Repository standing

| | Before | After |
|---|---|---|
| Tests | 81 | 104 |
| `qsv` modules | 10 | 11 |
| Notebooks | 14 | 15 |
| Commits ahead of GitHub | 19 | 22 |

Dataset regeneration still hashes to `bb787d3f020f2826410b8b97a9874965`, identical to
the value recorded at the translation pass: nothing upstream drifted. black clean on 22
files. Working tree clean.

## What is left

**The P0 code scope is closed.** No code item remains that the project's own documents
declare as a scientific limitation.

Open, in order:

1. **Adversarial invalid states** hugging the margin. This is now the main honesty gap:
   every accuracy figure in the repository describes our own generator's population,
   and the notebook-12 conclusion that higher dimension is easier is stated with that
   caveat attached. It is a code item and could be done without external dependencies.
2. **Web-app acceptance** (T1-T6 / V1-V3 in `reports/EMERGENT_ACCEPTANCE.md`), blocked
   on Emergent credits, then design phase 1.
3. **Tag v0.5.0**, deliberately gated on that acceptance.
4. **PyPI publication** - a manual step for the maintainer per `docs/RELEASING.md`;
   credentials are never automated in this repository.

Also pending on the maintainer's side: `git push origin main` (22 commits).
