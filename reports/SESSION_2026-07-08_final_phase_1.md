# Session report — Final phase, part 1: the project report notebook

**Date:** 2026-07-08

## Delivered

| Item | Content |
|---|---|
| `notebooks/13_project_report.ipynb` (EN, executed) | The full project report: presentation and objectives, theoretical foundations (Born rule, norm non-observability, noise model and its verified 2d*sigma^2 bias), methodological choices with rationale (experiments-before-text, F2 boundary, seed discipline, statistics-first), architecture, results table (one line per regime, each backed by an executed notebook), hypotheses and limits stated plainly (Gaussian simplification, synthetic invalid population, drift in-distribution caveat). |
| Section 7 — error log | First-person development logbook covering only real, documented incidents: the target leakage (context, the evening I believed the 100%, the failed quick fix, the reformulation, ~2 days), the six months of uncommitted work + CRLF churn + the gitignore negation that silently unversioned the dataset and hid tests/, the np.allclose hidden rtol (the validator lying about its own threshold by 10x), the two refuted hypotheses of notebooks 09-10 (published as negative results), and the smaller scars (notebook/module drift, numpy.bool vs JSON, sed vs CRLF, the environment loss that seeds turned into a 15-minute recovery). Honest time estimates on each. |
| Translation progress | `qsv/features.py` fully rewritten in English — the 57-test suite validates the transcription (invariance, bounds, correlation structure, drift law all locked). Remaining translation work tracked as an ordered checklist in ROADMAP.md. |

## Method note

The error log was written from `reports/` and the git history exclusively — every
incident, number and sequence of events in section 7 is traceable to a session report
or a commit. Nothing invented, as per the project conventions.

## Next

Remaining translation checklist (data_generation, preprocessing, README, notebooks
01-10 markdown), then tag v0.5.0 once the Emergent web app (milestone 5c, in progress)
is reviewed against qsv/validators.py. Progress: ~98.5%.

## Addendum — part 2 (same day): the whole codebase is now English

- `qsv/data_generation.py` translated with the strongest possible proof that the
  logic is untouched: the dataset regenerated with seed=42 before and after the
  translation has the **same MD5 hash** (bb787d3f...965).
- `qsv/preprocessing.py`, `qsv/paths.py`, `tests/conftest.py` and the three
  French-commented test files translated; 57 tests collected and green before and
  after; zero accented characters remain anywhere in `src/qsv/` or `tests/`.
- Remaining in the checklist: README.md, notebooks 01-06 (slim down + translate),
  notebooks 07-10 markdown, CONTRIBUTING/data-README/ROADMAP/CHANGELOG.

## Addendum — part 3: README fully in English

Full translation (580 lines): description with the honest ML framing, key-results
table, architecture tree (now listing all 13 notebooks), installation, the two usage
modes, theoretical foundations (including the Shannon-vs-von-Neumann precision), the
six aerospace application domains, the 25 references, milestone-status roadmap
summary, technologies, contributing summary, license and citation (year updated to
2026). Remaining in the checklist: notebooks 01-06 (slim down + translate), notebooks
07-10 markdown, CONTRIBUTING/data-README.
