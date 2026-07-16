# Quantum State Validator - Data

## Main dataset

**File:** `processed/quantum_states_10000.csv`

### Description

Synthetic quantum-state dataset for binary classification (valid/invalid).

### Characteristics

- **Size:** 10,000 samples (5,000 valid + 5,000 invalid)
- **Dimension:** 4 (Hilbert space of dimension 4)
- **Features:** 9 numeric columns + 1 binary target
- **Balance:** 50% class 0, 50% class 1

### Columns

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `state_id` | int | Unique identifier | 0 to 9999 |
| `c0_real` | float | Real part of c0 | R |
| `c0_imag` | float | Imaginary part of c0 | R |
| `c1_real` | float | Real part of c1 | R |
| `c1_imag` | float | Imaginary part of c1 | R |
| `c2_real` | float | Real part of c2 | R |
| `c2_imag` | float | Imaginary part of c2 | R |
| `c3_real` | float | Real part of c3 | R |
| `c3_imag` | float | Imaginary part of c3 | R |
| `norm_squared` | float | norm^2 = sum_i \|c_i\|^2 | [0, +inf) |
| `is_valid` | int | Label: 1=valid, 0=invalid | {0, 1} |

### Validity condition

A quantum state |psi> = (c0, c1, c2, c3) is **valid** if and only if:
```
norm^2 = |c0|^2 + |c1|^2 + |c2|^2 + |c3|^2 = 1
```

where |c_i|^2 = (real part)^2 + (imaginary part)^2.

### Generation strategies

**Valid states (`is_valid=1`):**
- Strategy: `random` (Gaussian generation + normalization)
- Guarantee: norm^2 = 1.000... (machine precision)

**Invalid states (`is_valid=0`):**
- Strategy: `mixed` (combination of several mechanisms)
  - 30% scaling (multiplication by k != 1)
  - 30% noise (additive noise without renormalization)
  - 30% direct (generation without normalization)
  - 10% extreme (pathological cases: near-null, huge, unbalanced)
- norm^2 range: [~0.0001, ~10]

### Reproducibility

- **Seed:** 42
- The full dataset regenerates identically with
  `create_dataset(n_valid=5000, n_invalid=5000, dim=4, seed=42)` -
  notebook `04__create_dataset.ipynb` is the executable documentation of
  this file (re-running it reproduces the CSV byte for byte).

### Usage

```python
import pandas as pd
from qsv.paths import MAIN_DATASET

df = pd.read_csv(MAIN_DATASET)

X = df.drop(["state_id", "is_valid"], axis=1)
y = df["is_valid"]
```

### Notes

- The `c{i}_real` / `c{i}_imag` features are independent (no strong correlation)
- `norm_squared` is **derived** from the others and *defines* the label:
  using it (or any bijection of it) as a training feature is target
  leakage - see the notebook 07 case study before touching it
- The dataset is shuffled (shuffle=True) to avoid any ordering bias
- No missing values

### Versions

- **v1.0** (2025-11-12): initial dataset, 10k samples, dim=4
- **v2.0** (2026-07-07): regenerated after the F2 boundary fix (see below),
  seed=42, byte-for-byte reproducible

---

## Class-boundary definition (F2 - audit 2026-07-07)

Since the F2 fix, the labeling rule is **unique and centralized** in
`generate_invalid_states` (parameter `norm_margin`, default 0.05):

| Class | Guarantee on norm^2 = sum_i \|c_i\|^2 |
|---|---|
| Valid (`is_valid = 1`) | norm^2 = 1 to machine precision (~1e-15) |
| Invalid (`is_valid = 0`) | \|norm^2 - 1\| >= `norm_margin` (forbidden band) |

**Reading**: no invalid state can lie within `norm_margin` of the unit norm.
The forbidden band (1 - margin, 1 + margin) embodies the physical ambiguity:
a state at norm^2 = 1.001 is indistinguishable from a rounding error and
therefore has no defensible label.

History: before this fix, `scaling` excluded k in [0.95, 1.05] but
`noise`/`direct` could produce states inside that zone (guarantee limited to
\|norm^2 - 1\| > 1e-4) - the boundary depended on the generation strategy.
Locked by `tests/test_data_generation.py::test_invalid_states_respect_norm_margin`.
