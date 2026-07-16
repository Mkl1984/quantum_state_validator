# QSV Ecosystem Strategy

*Strategic analysis of the Quantum State Validator as the seed of a broader
open-source platform. Written 2026-07-16, against the v0.5.0-dev codebase
(68 commits, 57 tests, 13 notebooks, library + API + web app). This document
takes positions; it is not a wish list. Honest caveats are marked as such.*

## 0. What QSV actually is (the positioning that everything follows from)

QSV is NOT a quantum simulator, a circuit framework, or a Qiskit competitor -
and it must never pretend to be. Its unique, defensible identity is narrower
and stronger: **the validation-first, honesty-first layer of the quantum
software stack**, built on three assets no similar project combines:

1. A **rigorous decision core** (`qsv.validators`): physics-derived statistics
   (bias-corrected norm test, gain/pointing two-channel monitor) with
   measurement-budget sizing curves - auditable, testable, framework-agnostic.
2. A **documented scientific method**: the target-leakage case study, the
   "experiments before interpretations" discipline, negative results published,
   Bayes limits proven - a living curriculum on how to do ML on
   physics-defined problems without fooling yourself.
3. A **dual delivery**: importable library and HTTP service sharing one logic,
   plus a pedagogical web layer.

Every opportunity below is evaluated against this identity. Anything that
dilutes it (building yet another circuit composer, yet another Bloch sphere
app) is explicitly deprioritized.

---

## 1. Use cases, tiered by fit

### Tier A - core fit (build for these first)

**University teaching (QM + ML courses).** Audience: instructors and students
in 2nd-4th year physics/CS. Problem: courses teach quantum formalism and ML
separately; nobody teaches the failure modes of applying ML to physics. Value:
QSV's notebooks 07-12 are a ready-made 6-session lab arc (leakage -> noise ->
correlated noise -> drift -> known targets -> sizing) with executed outputs and
a centralized doc notebook. Workflow: fork repo, run labs, grade against the
test suite. Required features: nbgrader-compatible assignment variants,
instructor answer keys. Limitation: pure states, single basis, d<=16 today.
Improvement: a course pack with graded exercises per notebook.

**Scientific reproducibility / methodology teaching.** Audience: research
methods courses, ML-rigor workshops. Problem: leakage and evaluation
malpractice are epidemic; teaching material with a REAL, fully documented
leakage incident (code, wrong 100% results preserved, fix, error log in the
author's voice) barely exists. Value: notebook 07 (archived with outputs) +
notebook 13 section 7 is the product. This is QSV's most differentiated asset.

**State-preparation validation / data QA for quantum pipelines.** Audience:
developers of simulators, tomography pipelines, quantum-chemistry codes.
Problem: numerical state vectors drift (normalization, gain) silently through
pipelines. Value: `validate_state` as a one-line guardrail; the CI-gate
pattern (see integrations) makes it a "pytest for state vectors". Workflow:
`pip install quantum-state-validator`, assert on outputs, or gate in CI.
Limitation: pure state vectors only; density matrices are the top roadmap item.

**Aerospace/satellite quantum technologies (pedagogical modeling).** Audience:
aerospace engineering programs, quantum-sensor teams onboarding juniors.
Problem: the budget-vs-error trade-off of embedded quantum instruments (GNSS
clocks, cold-atom IMUs, QKD) has no accessible teaching model. Value: QSV IS
that scale model - sizing curves, drift/recalibration study, FDIR-style cause
diagnosis. Honest caveat: it is a teaching model, not flight software, and
must always say so.

### Tier B - strong fit with modest work

**Bootcamps and online platforms.** The web app (playgrounds + lessons +
quizzes) is the vehicle; needs accounts/progress persistence and embeddable
widgets. **Hackathons/competitions**: the multiclass-cause and drift datasets
generate naturally into Kaggle-style challenges with a built-in
leakage trap - a genuinely novel competition design ("beat the threshold test
if you can"). **Quantum error detection / hardware benchmarking (entry
level)**: the norm/fidelity monitors generalize to acceptance testing of NISQ
readout data once a multinomial tomography model ships. **Academic
publications**: notebooks as executable supplementary material (see section 6).

### Tier C - adjacent (possible, not a priority)

VQE/QAOA and QML workflows can *consume* QSV as a state-sanity layer
(validate ansatz outputs each iteration) - integration, not core development.
High-school outreach works through the web app's lessons 1-2 only (the rest
assumes linear algebra). Quantum communication/cryptography: the QKD analogy
is pedagogically sound; a real QBER module would be a new subproject.

### Tier D - honest misfits (do not chase)

Quantum chemistry and materials science as *domains*: QSV can guard their
pipelines (Tier A, data QA) but has nothing domain-specific to offer; claiming
otherwise would damage credibility. Circuit debugging in the full sense
(gate-level) belongs to Qiskit/Cirq tooling.

---

## 2. Integrations (architecture: one adapter layer, one decision core)

The single architectural rule: **every integration is a thin adapter that
converts a foreign state representation into (real[], imag[]) and calls
`qsv.validators`**. No integration reimplements logic - the web-app
acceptance protocol (reports/EMERGENT_ACCEPTANCE.md) exists precisely because
one implementation drifted from the reference; adapters make that structural.

| Target | Adapter | Data flow | Complexity |
|---|---|---|---|
| **Qiskit** | `qsv.adapters.qiskit`: `Statevector` -> validate; `Counts` -> multinomial estimates (needs the tomography module) | `Statevector.data` (complex ndarray) -> re/im lists -> ValidationResult | Low (statevector), medium (counts) |
| **PennyLane** | `qml.state()` output -> validate; per-step callback in optimization loops (VQE sanity gate) | complex ndarray each iteration | Low |
| **Cirq** | `cirq.final_state_vector` -> validate | same | Low |
| **Braket / Azure Quantum** | result-JSON parsers -> amplitudes or counts -> REST call to qsv.api (cloud-side) or local lib | JSON over HTTPS | Medium (auth, formats) |
| **TensorFlow Quantum / PyTorch** | dataset loaders: `create_dataset`/`create_multiclass_dataset` as `tf.data`/`Dataset` factories; leakage-safe feature split preserved in the loader API | ndarray/tensors | Low |
| **JupyterLab** | native today; add a rich `_repr_html_` on ValidationResult (verdict card) | in-process | Trivial |
| **VS Code extension** | wraps the REST API: validate the state under cursor / a .npy file | HTTP to local uvicorn | Medium |
| **GitHub Actions** | `qsv-validate` action: fail CI if committed state files / simulation outputs break validity - the "pytest for state vectors" wedge | file glob -> exit code | Low, HIGH leverage |
| **Docker** | `docker run qsv-api` (uvicorn image); compose file with the web app | - | Low |
| **FastAPI** | done (qsv.api) | - | Done |
| **Next.js/React/Three.js** | done (web app); Three.js only for the d=2 Bloch section per the scientific constraint | REST | Done/ongoing |
| **SciPy ecosystem** | already native (numpy/pandas in, dataclasses out) | - | Done |

Priority order: GitHub Action > Qiskit/PennyLane adapters > dataset loaders >
the rest. The Action is the adoption wedge: it meets developers where they
already are, costs a day, and every CI badge is advertising.

---

## 3. Similar projects - lessons

| Project | What it is | Lesson for QSV |
|---|---|---|
| **QuTiP** | The reference open-source quantum dynamics toolkit | Longevity comes from being a *library first* with impeccable docs; QSV's lib/API split follows this. Gap QSV fills: QuTiP computes, it does not *teach decision-making under noise*. |
| **Qiskit (textbook + visualization)** | Industry framework + education arm | Don't compete on breadth; integrate. Their textbook lacks a validation/rigor track - that is QSV's slot. |
| **Quirk** (quantum circuit simulator in the browser) | Beloved drag-and-drop circuit toy | Instant, install-free interactivity drives adoption; the QSV web playgrounds follow this. Quirk has zero methodology layer. |
| **PhET simulations** | Gold standard of physics education sims | Pedagogical polish and guided discovery beat feature count. Also: institutional backing matters for reach. |
| **nbgrader** | Jupyter assignment/grading | Integrate rather than rebuild - QSV course pack should be nbgrader-native. |
| **MLflow / W&B** | Experiment tracking | The "experiments before interpretations" discipline could be enforced by tooling - a lightweight run-registry for the notebooks is a future module, not a company. |
| **SymPy** | Symbolic math | Docstring culture ("reading" of each formula) parallels SymPy's; a symbolic verification of the 2d*sigma^2 bias would make a nice doc notebook cell. |
| **Great Expectations** | Data-quality assertions for data engineering | The closest *conceptual* sibling: QSV is Great Expectations for quantum states. Their success proves validation-as-a-product works; their DSL-heaviness is the anti-pattern to avoid. |

---

## 4. Complementary projects (integration targets, prioritized)

High value, shared users, cheap integration: **state tomography tools** (feed
their reconstructions into QSV; QSV's noise model becomes their error bar),
**quantum experiment notebooks** (QSV as the validation cell), **probability
amplitude visualizers** (embed QSV verdicts). Medium: Bloch sphere visualizers
(d=2 section only), density-matrix explorers (after the rho module),
linear-algebra/eigenvalue labs and complex-number visualizers (prerequisite
links from lessons - curate, don't build), Fourier/wavefunction simulators
(link out). Low/decline: error-correction visualizers and quantum-optics
simulators (different scope; revisit post-v2). Rule: **curate and link
prerequisites; only build what touches validation.**

---

## 5. Future QSV modules (prioritized)

| Module | What | Value | Difficulty | Priority |
|---|---|---|---|---|
| `qsv.tomography` | Multinomial finite-shot model (per-setting counts -> estimates), replacing the Gaussian simplification | Closes the main stated limitation; unlocks hardware-counts adapters | Medium | **P0** |
| `qsv.density` | Density-matrix validity: Hermiticity, unit trace, positivity (eigenvalue check) + noisy versions | Opens mixed states, error channels; doubles the theory surface | Medium | **P0** |
| `qsv.adapters` | Qiskit/PennyLane/Cirq/counts converters | Adoption wedge | Low | **P0** |
| CLI (`qsv validate file.npy`) + GitHub Action | CI guardrail | Adoption wedge | Low | **P0** |
| Course pack (nbgrader) | Assignments + rubrics per notebook | Teaching adoption | Medium | P1 |
| Drift monitor | Streaming recalibration (notebook 10 productized): rolling estimator + alarm API | Research/industrial value | Medium | P1 |
| Adversarial generator | Margin-hugging invalid states (stress the sizing curves) | Scientific honesty upgrade | Low | P1 |
| Challenge server | Timed missions, leaderboards (web app phase 3) | Engagement | High | P2 |
| Entanglement module | Schmidt decomposition, separability checks for d=4 as 2x2 qubits | Big pedagogy win, heavier math | High | P2 |

---

## 6. Research ecosystem

The realistic research role is **infrastructure, not results**: (a) executable
supplementary material - a paper's state-preparation claims shipped as a QSV
notebook whose acceptance tests readers re-run (the EMERGENT_ACCEPTANCE.md
pattern generalized: "acceptance protocols" as first-class artifacts);
(b) validation gates in research-code CI (reproducibility by construction -
QSV's own byte-identical dataset regeneration is the demo); (c) teaching labs
in methods courses; (d) an open-science template: the audit -> error-log ->
negative-results-published workflow is itself citable methodology. Concrete
first step: a JOSS (Journal of Open Source Software) submission - QSV fits
JOSS's scope exactly and the review would harden the package.

## 7. Educational ecosystem

Layered on the web app: guided lessons (exists) -> adaptive quizzes (exists,
make item difficulty responsive to answers) -> guided experiments (playgrounds
with parameter missions: "find the minimal N for FPR<1%") -> challenge mode
(planned phase 3) -> practical labs (the notebooks via JupyterLite in-browser,
zero install) -> assignments & grading (nbgrader course pack; the 57-test
suite doubles as an autograder for code assignments) -> instructor dashboard
(class progress, per-lesson error patterns - needs accounts; v2.0 scope).
Principle carried over from the codebase: **every wrong quiz answer gets a
"why", and every displayed number traces to an implemented formula.**

## 8. Technical roadmap

| Phase | Content | Users |
|---|---|---|
| **MVP (= today, v0.5.0 pending web-app acceptance)** | Library + API + 13 notebooks + web app v1; CI; docs | Author, early adopters, recruiters |
| **v1.0** | P0 modules: tomography, density matrices, adapters, CLI + GitHub Action; PyPI release; JOSS paper; adversarial generator | Python quantum devs, instructors |
| **v2.0** | Course pack (nbgrader), accounts + instructor dashboard, JupyterLite labs, drift-monitor API, challenge mode | Universities, bootcamps |
| **v3.0** | Entanglement module, hardware-counts benchmarking suite, community challenge server, plugin API for third-party validators | Research groups, community |
| **Long term** | The "Great Expectations of quantum computing": the neutral validation layer every framework calls | Ecosystem-wide |

Architecture changes: v1.0 splits `qsv` into `qsv-core` (zero heavy deps) +
extras; v2.0 adds a persistence service behind the web app; v3.0 defines a
validator plugin interface.

## 9. Competitive analysis (candid)

Nobody occupies "validation-first quantum pedagogy" - that is the good news
and the warning: the niche is open partly because it is narrow. Competitors by
overlap: Qiskit Textbook (breadth, brand; no rigor/validation track - QSV
wins on methodology depth and honest failure documentation), QuTiP (numerical
power; no pedagogy of decision-making), Quirk/PhET (interactivity polish; no
library, no science-under-noise), Great Expectations (the model to emulate, in
another domain). QSV already does better than all of them on exactly three
things: documented negative results, measurement-budget engineering curves,
and the lib/API/no-divergence architecture. Gaps to close before claiming the
niche publicly: density matrices, multinomial model, PyPI presence, and at
least one external adopter. Differentiation to protect: the error-log culture
- it is unfakeable and no incumbent will copy it.

## 10. Innovation opportunities (feasible, prioritized)

1. **The Leakage Hunt** (unique, cheap, viral potential): an interactive game
   where players inspect a feature set and a perfect confusion matrix and must
   find the leak before the timer - generated from QSV's own feature
   machinery. No equivalent exists anywhere.
2. **Budget-sizing simulator as an engineering tool**: expose the notebook 12
   abacus as an interactive "mission designer" (choose margin, FPR spec, get
   N and integration-time analogies). Bridges education and real sizing.
3. **AI tutor grounded in the validators**: an LLM assistant whose every
   numeric claim is computed by qsv (never generated) - the architecture
   pattern QSV already enforces for its web app, extended to dialogue.
4. **Collaborative lab sessions**: shared playground state via URL (already
   half-designed in the web app's sharing feature) - two students, one state,
   argue about the verdict.
5. **Acceptance protocols as a genre**: publish the EMERGENT_ACCEPTANCE.md
   format as a mini-spec ("how to certify a third-party reimplementation of a
   scientific library") - a small, citable, genuinely novel contribution.

---

*Next concrete actions, in order: (1) unblock the web-app acceptance and tag
v0.5.0; (2) PyPI release; (3) qsv.adapters + GitHub Action; (4) qsv.density +
qsv.tomography; (5) JOSS submission.*
