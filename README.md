# Quantum State Validator

**Quantum State Classification with Machine Learning**

[![CI](https://github.com/Mkl1984/quantum_state_validator/actions/workflows/ci.yml/badge.svg)](https://github.com/Mkl1984/quantum_state_validator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-57%20passed-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Description

This project implements a complete machine-learning pipeline to automatically classify the validity of discrete quantum states, combining quantum-mechanics principles with statistical learning techniques.

### The physics problem

In quantum mechanics, a pure state |psi> is a vector in a finite-dimensional Hilbert space. To be physically valid, the state must satisfy the **normalization condition** (Born rule):

$$\sum_{i=1}^{n} |c_i|^2 = 1$$

where the $c_i \in \mathbb{C}$ are probability amplitudes and $|c_i|^2$ is the probability of measuring the state in basis state $|i\rangle$.

### The machine-learning challenge

On **exact** amplitudes, validity is a deterministic function of the features - $\|\psi\|^2$ can simply be computed, there is nothing to learn. The problem is then either trivial (with any scale-sensitive feature) or impossible (with only scale-invariant features). Notebook 07, deliberately kept as a **target-leakage case study**, documents this dead end: excluding `norm_squared` while keeping `norm_deviation = |norm_squared - 1|` produced 100% accuracy... by training the model on the label definition.

**The honest formulation** (notebook 08): amplitudes are observed through **finite-shot tomographic noise** with budget N, i.e. $\hat{c}_i = c_i + \varepsilon$ with $\sigma = 1/(2\sqrt{N})$ - the shot-noise law. Near the boundary $\|\psi\|^2 = 1$ the classes genuinely overlap: deciding becomes a legitimate statistical problem, with real false-positive/false-negative trade-offs driven by the budget N. This is exactly the operational problem of an **embedded quantum-state qualification system** (see the aerospace applications section).

---

## Key Results

Notebook 08 - test accuracy versus measurement budget N (10,000 states, dim = 4, seed = 42):

| N (shots) | Bias-corrected threshold test (1 parameter) | Logistic regression | Random Forest (100 trees) |
|---|---|---|---|
| 50 | **0.913** | 0.603 | 0.906 |
| 500 | **0.972** | 0.602 | 0.974 |
| 5,000 | **0.997** | 0.606 | 0.998 |

Three lessons:

1. **The one-parameter statistical test ties the Random Forest**: all the validity information lives in the estimated norm $\|\hat{\psi}\|^2$, as the physics dictates. Understanding the structure of the problem beats stacking models.
2. **Performance is driven by the physics** (the budget N), not by the algorithm. The engineering question becomes: what minimum N for a contractual false-positive rate?
3. **The logistic regression collapses (~60%)**: the valid class lives inside a *band* of norms - a geometry that is not linearly separable - made worse by heavy-tailed features. Representation matters more than the algorithm.

The measured norm-estimator bias ($E[\|\hat{\psi}\|^2] - \|\psi\|^2 = 0.0398$ at N = 50) matches the theoretical prediction $2d\sigma^2 = 0.0400$ - the noise model behaves exactly as theory requires.

---

## Learning Objectives

### ML skills developed

- Synthetic data generation under physical constraints (Dirichlet, controlled perturbations)
- Specialized feature engineering (Shannon entropy of the Born distribution, quantum purity)
- Complete ML pipeline: preprocessing, training, validation, interpretation
- Rigorous evaluation: confusion matrices, ROC curves, learning curves
- Statistical decision theory as a baseline for ML claims
- Advanced visualization: interactive 3D Plotly, geometric analyses
- Best practices: Git versioning, modular structure, documentation, tests, CI

### Physics concepts applied

- Born rule and quantum normalization
- Discrete Hilbert spaces and the probability simplex
- Shannon entropy of the measurement distribution and purity measures
- Random state generation on the simplex (Dirichlet distribution)
- Shot noise, estimator bias, measurement-budget sizing

---

## Project Architecture

```
quantum_state_validator/
|
|-- .github/workflows/ci.yml     # CI: black + pytest (Python 3.10 / 3.12)
|-- data/
|   |-- raw/
|   |-- processed/
|   |   |-- quantum_states_10000.csv   # 10,000 states, dim=4, seed=42 (reproducible)
|   |-- README.md                # Schema + labeling rule (F2 boundary)
|-- notebooks/
|   |-- 01_quantum_theory.ipynb        # Theory: normalization, Born rule
|   |-- 02_test_data_generation.ipynb  # Valid-state generation (3 strategies)
|   |-- 03_test_invalid_states.ipynb   # Invalid-state generation (4 strategies)
|   |-- 04__create_dataset.ipynb       # Dataset construction
|   |-- 05_eda_advanced.ipynb          # Advanced EDA (3D Plotly, simplex)
|   |-- 06_test_preprocessing.ipynb    # Preprocessing pipeline
|   |-- 07_model_evaluation.ipynb      # ARCHIVED: target-leakage case study
|   |-- 08_measurement_noise.ipynb     # Honest formulation: measurement noise
|   |-- 09_correlated_noise_multiclass.ipynb  # Common-mode noise + cause diagnosis
|   |-- 10_calibration_drift.ipynb     # Non-stationary regime: hybrid wins
|   |-- 11_preparation_qa.ipynb        # Known-target QA: isotropy limit broken
|   |-- 12_scaling_and_sizing.ipynb    # Dimensions + N-vs-error sizing curves
|   |-- 13_project_report.ipynb        # Full project report + error retrospective
|-- src/qsv/                     # Installable package (pip install -e .)
|   |-- validators.py            # Pure decision logic - the library mode
|   |-- api.py                   # FastAPI HTTP service - thin wrapper
|   |-- data_generation.py       # Valid/invalid states, F2 boundary guarantee
|   |-- features.py              # Invariant vs sensitive features + noise models
|   |-- preparation.py           # Known-target preparation QA
|   |-- paths.py                 # Repository paths (notebooks)
|   |-- preprocessing.py         # Stratified 60/20/20 split + leak-free scaling
|-- tests/                       # 57 pytest tests (anti-leakage, F2 guarantee, splits)
|-- reports/                     # Full audit + session reports
|-- CHANGELOG.md . CONTRIBUTING.md . LICENSE . ROADMAP.md
|-- requirements.txt . pyproject.toml
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- 4 GB RAM minimum (8 GB recommended)

### Steps

#### 1. Clone the repository

```bash
git clone https://github.com/Mkl1984/quantum_state_validator.git
cd quantum_state_validator
```

#### 2. Create a virtual environment

**With venv (Windows):**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**With venv (macOS/Linux):**

```bash
python -m venv venv
source venv/bin/activate
```

**With conda:**

```bash
conda create -n qsv_env python=3.11 -y
conda activate qsv_env
```

#### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.14.0
- jupyter >= 1.0.0

#### 4. Install the package and verify

```bash
pip install -e ".[api,dev]"
python -c "import qsv; print('qsv', qsv.__version__)"
pytest tests/ -q        # the whole suite must pass
```

The same verification runs in continuous integration (GitHub Actions) on every push: `black --check` formatting, then the `pytest` suite on Python 3.10 and 3.12.

---

## Usage - two distinct, combinable modes

QSV can be used **without the application**, as a regular Python library, or as an HTTP service - both modes share exactly the same decision logic (`qsv/validators.py`), distilled from notebooks 08-12.

### Mode 1: library (inside your own project)

```python
from qsv import validate_state, preparation_qa

# Exact amplitudes: validity is computed, not inferred
result = validate_state(real=[1.0, 0, 0, 0], imag=[0.0, 0, 0, 0])
print(result.valid, result.explanation)

# Finite-shot tomography data: bias-corrected threshold test,
# with a budget-sufficiency flag (notebook 12 sizing curve)
result = validate_state(real, imag, n_shots=500, margin=0.05)

# Known-target preparation QA: two-channel gain/pointing monitor
result = preparation_qa(real, imag, target_real, target_imag)
print(result.error_type)   # "ok" | "gain_error" | "pointing_error" | ...
```

### Mode 2: HTTP service (from any language)

```bash
uvicorn qsv.api:app --reload
# then http://localhost:8000/docs (interactive OpenAPI documentation)
```

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"real": [1.0, 0, 0, 0], "imag": [0.0, 0, 0, 0], "n_shots": 500}'
```

### Combined

The service is a client of the library: a Python pipeline can import `qsv` directly while a remote application queries the same logic over HTTP - the two verdicts cannot diverge, by construction.

---

## Theoretical Foundations

### Quantum mechanics

#### Hilbert space and pure states

A pure quantum state in dimension $n$ is a vector $|\psi\rangle$ of the Hilbert space $\mathcal{H} \cong \mathbb{C}^n$:

$$|\psi\rangle = \sum_{i=0}^{n-1} c_i |i\rangle$$

where $\{|i\rangle\}_{i=0}^{n-1}$ is an orthonormal basis.

#### Born rule

The probability of measuring state $|i\rangle$ is:

$$P(i) = |c_i|^2 = |\langle i | \psi \rangle|^2$$

#### Normalization condition

For the probabilities to be consistent (sum = 1):

$$\langle \psi | \psi \rangle = \sum_{i=0}^{n-1} |c_i|^2 = 1$$

This condition defines the **unit sphere** in $\mathbb{C}^n$.

#### The probability simplex

Considering only the probabilities $p_i = |c_i|^2$, valid states form a **simplex**:

$$\Delta^{n-1} = \left\{ (p_0, \ldots, p_{n-1}) \in \mathbb{R}^n : p_i \geq 0, \sum_i p_i = 1 \right\}$$

**Geometric properties:**
- Dimension: $n-1$
- Boundary: the hyperplane $\sum_i p_i = 1$
- 3D example (n=4): a regular tetrahedron

### Quantum features

The `qsv/features.py` module splits the features into two families, and this split is **the methodological core of the project**:

- **Scale-invariant** (computed on $\tilde{p}_i = p_i / \sum_j p_j$): insensitive to $c \to k c$, hence structurally blind to validity. A classifier using only them caps at ~50% - that is a theorem, not a failure.
- **Scale-sensitive** (computed on the raw $p_i$): they encode the norm, hence the label. On exact data they are leaks; on noisy data (notebook 08) they are legitimate estimators.

This property is locked by a mechanical test (`tests/test_features.py::test_invariant_features_are_scale_invariant`).

#### 1. Shannon entropy of the Born distribution

$$H(\tilde{p}) = -\sum_{i=0}^{n-1} \tilde{p}_i \ln(\tilde{p}_i)$$

**Interpretation:**
- $H = 0$: pure basis state; $H = \ln(n)$: uniform superposition
- Measures the spread of the state over the measurement basis

**Physics note**: for a *pure* state, the von Neumann entropy $S(\rho) = -\mathrm{Tr}(\rho \ln \rho)$ is exactly 0. The useful quantity here is the Shannon entropy of the measurement distribution $\tilde{p}_i$ (Born rule) in the computational basis - hence the name `entropy_shannon` in the code (in nats: natural logarithm; conversion to bits: factor $1/\ln 2$).

#### 2. Quantum purity

$$P(\psi) = \sum_{i=0}^{n-1} p_i^2$$

**Properties:**
- $P = 1$: pure basis state
- $P = 1/n$: maximally spread state
- Relation to entropy: $P$ and $H$ are inversely correlated
- Trace link: $P = \text{Tr}(\rho^2)$ where $\rho = |\psi\rangle\langle\psi|$

#### 3. Norm deviation

$$D(\psi) = \left| \sum_{i=0}^{n-1} |c_i|^2 - 1 \right|$$

**Interpretation:**
- $D = 0$: valid state
- $D > 0$: invalid state

**Warning**: on exact amplitudes this quantity *defines* the label - using it as a feature IS target leakage (the notebook 07 lesson). On noisy amplitudes (notebook 08), its estimated version becomes the legitimate decision statistic, to be corrected for the $2d\sigma^2$ bias.

### Theorems used

**Theorem (Riesz-Fischer)**: the Hilbert space $\mathcal{H}$ is complete for the norm induced by the inner product.

**Theorem (Spectral)**: for a Hermitian operator $A$, there exists an orthonormal basis of eigenvectors.

**Lemma (Cauchy-Schwarz)**: for all $|\psi\rangle, |\phi\rangle \in \mathcal{H}$:

$$|\langle \psi | \phi \rangle|^2 \leq \langle \psi | \psi \rangle \cdot \langle \phi | \phi \rangle$$

---

## Applications - Aerospace and Aeronautics

This project is a **pedagogical scale model of the state-qualification subsystem** carried by every embedded quantum sensor. The common thread: a quantum instrument in flight never knows its exact amplitudes - it must decide the conformity of a state from **finite measurement statistics**, under resource constraints. That is exactly the structure of notebook 08.

### 1. Atomic clocks for satellite navigation (GNSS)

The GPS (rubidium RAFS clocks) and Galileo (passive hydrogen masers PHM + RAFS) constellations rely on interrogating atomic states. Frequency stability (Allan variance) improves as 1/sqrt(N) with the number of interrogations - the same law as our sigma = 1/(2 sqrt(N)). The notebook 08 dilemma (budget N vs false-positive rate) is the sizing problem of a clock's integration time against a stability specification, on which ground positioning error directly depends (1 ns of clock error ~ 30 cm of pseudo-range error).

### 2. Quantum inertial navigation (atom interferometry)

Cold-atom gyroscopes and accelerometers (ONERA work - the marinized GIRAFE gravimeter, quantum inertial navigation projects for GNSS-denied navigation) require conformal atomic state preparation before each interferometry cycle. In a vibrating environment (an aircraft carrier platform), qualifying the prepared state from partial measurements is a critical step: a badly prepared state accepted = a navigation bias; a correct state rejected = lost measurement cadence. That is the FP/FN trade-off of our ROC curve.

### 3. Satellite quantum key distribution (QKD)

The Micius satellite (2016, 1,200 km ground-space QKD link) and the European EuroQCI/Eagle-1 programs qualify photonic states (polarization, entanglement) from finite samples: the quantum bit error rate (QBER) estimated on a measurement subset decides whether the key is safe or discarded. Photon budget = our budget N; QBER threshold = our threshold on the estimated norm.

### 4. Quality control of quantum computers for aerospace optimization

The use cases studied by manufacturers (trajectory optimization, payload scheduling, fleet logistics - Airbus ran a dedicated Quantum Computing Challenge) assume correctly prepared qubit registers. **Verification tomography** between two computation campaigns is a state-validation-under-budget problem - sizing that budget is our central question.

### 5. Quality assurance of quantum numerical simulations

Electronic-structure computations for aerospace materials (alloys, catalysts, batteries) produce numerical state vectors whose norm drifts with rounding errors and integration schemes. A QSV-type validator plays the role of a **pipeline guardrail** (data QA): catching a normalization drift before it contaminates the computed observables.

### 6. Embedded quantum magnetometry

Optically pumped and NV-center magnetometers for magnetic anomaly detection (map-matching navigation, airborne prospecting) rely on spin states whose preparation quality drives sensitivity. Embedded diagnosis = statistical decision on finite measurements, at a cadence imposed by the carrier dynamics.

**Cross-cutting reading**: in all six cases, the engineering quantity is not "model accuracy" but the pair (measurement budget N, tolerated error rate) - and the project's lesson is that a well-built statistical test on the right statistic often suffices, ML adding value only under structured noise (correlations, calibration drift: see ROADMAP milestone 4).

---

## References

### Quantum mechanics

1. **Cohen-Tannoudji, C., Diu, B., & Laloe, F.** (2019). *Mecanique quantique* (Vols. 1 & 2). EDP Sciences.
   - Comprehensive French-language reference, rigorous treatment of Hilbert spaces

2. **Basdevant, J.-L., & Dalibard, J.** (2002). *Mecanique quantique*. Editions de l'Ecole Polytechnique.
   - French pedagogical approach, excellent for the foundations

3. **CNRS - Introduction a la mecanique quantique** (2020). *Part 1: Postulates and formalism*. Online course.
   - Available at: https://www.cnrs.fr/fr/

4. **CNRS - Introduction a la mecanique quantique** (2020). *Part 2: Measurement and evolution*. Online course.
   - Companion to part 1

5. **Nielsen, M. A., & Chuang, I. L.** (2010). *Quantum Computation and Quantum Information* (10th Anniversary Edition). Cambridge University Press.
   - The international reference, chapters 2-3 on postulates and measurement

6. **Sakurai, J. J., & Napolitano, J.** (2017). *Modern Quantum Mechanics* (2nd Edition). Cambridge University Press.
   - Chapter 1: Hilbert spaces and quantum states

7. **Griffiths, D. J., & Schroeter, D. F.** (2018). *Introduction to Quantum Mechanics* (3rd Edition). Cambridge University Press.
   - Excellent for physical intuition

8. **Le Bellac, M.** (2013). *Physique quantique* (2nd edition). EDP Sciences.
   - Modern approach with current applications

### Machine learning

9. **Geron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition). O'Reilly.
   - Chapter 3: Classification; Chapter 6: Decision trees and Random Forests

10. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
    - Section 1.5: Decision theory; mathematically rigorous treatment

11. **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2021). *An Introduction to Statistical Learning* (2nd Edition). Springer.
    - Chapter 4: Classification; very accessible with R/Python examples

12. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd Edition). Springer.
    - Advanced reference for statistical theory

### Scientific articles

13. **Fawcett, T.** (2006). "An introduction to ROC analysis". *Pattern Recognition Letters*, 27(8), 861-874.
    - Complete reference on ROC curves and AUC

14. **Breiman, L.** (2001). "Random Forests". *Machine Learning*, 45(1), 5-32.
    - The original Random Forests paper

15. **Lundberg, S. M., & Lee, S. I.** (2017). "A Unified Approach to Interpreting Model Predictions". *Advances in Neural Information Processing Systems*, 30.
    - Theoretical foundations of SHAP

16. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier". *Proceedings of KDD*, 1135-1144.
    - Introduction to LIME (Local Interpretable Model-agnostic Explanations)

### Quantum machine learning

17. **Schuld, M., & Petruccione, F.** (2018). *Supervised Learning with Quantum Computers*. Springer.
    - Bridge between classical and quantum ML

18. **Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., & Lloyd, S.** (2017). "Quantum machine learning". *Nature*, 549(7671), 195-202.
    - State-of-the-art QML review

### Technical documentation

19. **Scikit-learn Documentation**: https://scikit-learn.org/stable/
    - Metrics module: https://scikit-learn.org/stable/modules/model_evaluation.html

20. **Plotly Documentation**: https://plotly.com/python/
    - 3D visualizations: https://plotly.com/python/3d-charts/

21. **NumPy Documentation**: https://numpy.org/doc/stable/
    - Reference for numerical computing

### Online resources

22. **Quantum Country**: https://quantum.country/
    - Interactive introduction to quantum mechanics

23. **3Blue1Brown**: https://www.youtube.com/c/3blue1brown
    - "Essence of Linear Algebra" series; outstanding visualizations for vector spaces

24. **StatQuest (Josh Starmer)**: https://www.youtube.com/c/joshstarmer
    - Random Forests and ML metrics series; clear and rigorous pedagogy

25. **College de France - Alain Aspect**: https://www.college-de-france.fr/
    - "Physique quantique" lectures (2020-2021); advanced but accessible

---

## Project Roadmap

Living, detailed roadmap: see [ROADMAP.md](ROADMAP.md). Summary:

### Milestone 0: Setup - DONE
Environment, professional project structure, Git, dependencies.

### Milestone 1: Theory and data (v0.1.0) - DONE
Quantum theory notebooks, `data_generation` module, generation strategies, 10,000-sample dataset.

### Milestone 2: EDA and preprocessing (v0.2.0) - DONE
Advanced EDA (3D Plotly), quantum feature engineering, stratified 60/20/20 split.

### Milestone 3: Honest evaluation (v0.3.0) - DONE
- Notebook 07: first evaluation, invalidated by target leakage - **archived as a case study**
- Measurement-noise reformulation (notebook 08): ROC per budget N, the $2d\sigma^2$ bias, comparison to the optimal statistical test
- Class-boundary guarantee (norm_margin) + reproducible regenerated dataset
- 57 pytest tests (mechanical anti-leakage check included) + GitHub Actions CI

### Milestone 4: Where ML earns its place (v0.4.0) - DONE
Correlated noise (negative result owned), calibration drift (hybrid physics+ML wins), known-target preparation QA (isotropy limit broken), dimension scaling and N-vs-error sizing curves.

### Milestone 5: Production and interface (v1.0.0) - IN PROGRESS
Installable package + REST API done; interactive pedagogical web app in progress.

---

## Technologies

### Languages and frameworks

| Technology | Version | Usage |
|------------|---------|-------|
| Python | 3.10+ | Main language |
| NumPy | 1.24+ | Numerical computing |
| Pandas | 2.0+ | Data manipulation |
| Scikit-learn | 1.3+ | ML algorithms |
| Matplotlib | 3.7+ | Static visualizations |
| Seaborn | 0.12+ | Statistical visualizations |
| Plotly | 5.14+ | Interactive 3D visualizations |
| Jupyter | 1.0+ | Interactive notebooks |
| FastAPI | 0.110+ | Prediction API |

### Development tools

- IDE: VS Code with Python and Jupyter extensions
- Version control: Git + GitHub (Conventional Commits, release tags)
- Environment: venv or conda
- Quality: black (formatting), pytest (57 tests), GitHub Actions (CI on 3.10/3.12)
- Type annotations live directly in the modules

---

## Contributing

Full guide: [CONTRIBUTING.md](CONTRIBUTING.md). The essentials:

### Branching strategy

```bash
main              # Stable code
feature/name      # New features
fix/name          # Bug fixes
```

### Commit convention

Format: `<type>(<scope>): <subject>` (Conventional Commits, in English)

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`

```bash
git commit -m "feat(features): add correlated measurement noise"
git commit -m "fix(preprocessing): guard entropy against p=0"
git commit -m "docs: update README with notebook 08 results"
```

---

## Contact

**Author**: Mklzenin
Aerospace engineering student (AI and cybersecurity track)

**GitHub**: https://github.com/Mkl1984/quantum_state_validator

For questions or suggestions:
- GitHub Issues: https://github.com/Mkl1984/quantum_state_validator/issues
- Discussions: https://github.com/Mkl1984/quantum_state_validator/discussions

---

## License

Born from personal curiosity about the quantum world, this project grew into a genuine learning platform. Its goal is to offer a clear, intuitive path into the basic principles of quantum mechanics without sacrificing precision.

**MIT License**

Copyright (c) 2024 Mandem

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See the canonical [LICENSE](LICENSE) file.

---

## Citation

If you use this code in your work, please cite:

```bibtex
@misc{quantum_state_validator_2026,
  author = {Mklzenin},
  title = {Quantum State Validator: ML Classification of Quantum States},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Mkl1984/quantum_state_validator}}
}
```

---

## Perspectives

### Possible extensions

- Multinomial tomography model (replacing the Gaussian simplification)
- Adversarial invalid states hugging the margin
- Mixed states (density matrices): validity becomes positivity + unit trace
- Out-of-distribution calibration drift
- Deep learning and quantum ML (variational circuits)
- Real experimental data

### Applications

See the detailed [Applications - Aerospace and Aeronautics](#applications---aerospace-and-aeronautics) section.

---

Last update: July 2026
Version: 0.5.0-dev - see [CHANGELOG.md](CHANGELOG.md)
