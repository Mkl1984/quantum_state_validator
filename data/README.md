# Données du Projet Quantum State Validator

## Structure

## Dataset Principal

**Fichier :** `processed/quantum_states_dataset.csv`

### Description

Dataset synthétique d'états quantiques pour classification binaire (valide/invalide).

### Caractéristiques

- **Taille :** 10 000 échantillons (5000 valides + 5000 invalides)
- **Dimension :** 4 (espace de Hilbert de dimension 4)
- **Features :** 9 colonnes numériques + 1 target binaire
- **Équilibrage :** 50% classe 0, 50% classe 1

### Colonnes

| Colonne | Type | Description | Plage |
|---------|------|-------------|-------|
| `state_id` | int | Identifiant unique | 0 à 9999 |
| `c0_real` | float | Partie réelle de c₀ | ℝ |
| `c0_imag` | float | Partie imaginaire de c₀ | ℝ |
| `c1_real` | float | Partie réelle de c₁ | ℝ |
| `c1_imag` | float | Partie imaginaire de c₁ | ℝ |
| `c2_real` | float | Partie réelle de c₂ | ℝ |
| `c2_imag` | float | Partie imaginaire de c₂ | ℝ |
| `c3_real` | float | Partie réelle de c₃ | ℝ |
| `c3_imag` | float | Partie imaginaire de c₃ | ℝ |
| `norm_squared` | float | ||ψ||² = Σᵢ|cᵢ|² | [0, +∞) |
| `is_valid` | int | Label: 1=valide, 0=invalide | {0, 1} |

### Condition de Validité

Un état quantique |ψ⟩ = (c₀, c₁, c₂, c₃) est **valide** si et seulement si :
```
||ψ||² = |c₀|² + |c₁|² + |c₂|² + |c₃|² = 1
```

où |cᵢ|² = (partie_réelle)² + (partie_imaginaire)²

### Stratégies de Génération

**États valides (`is_valid=1`) :**
- Stratégie : `random` (génération gaussienne + normalisation)
- Garantie : ||ψ||² = 1.000... (précision machine)

**États invalides (`is_valid=0`) :**
- Stratégie : `mixed` (combinaison de plusieurs méthodes)
  - 30% scaling (multiplication par k ≠ 1)
  - 30% noise (ajout de bruit sans renormalisation)
  - 30% direct (génération sans normalisation)
  - 10% extreme (cas pathologiques : quasi-nuls, énormes, déséquilibrés)
- Plage de ||ψ||² : [0.0001, ~10]

### Reproductibilité

- **Seed :** 42
- Tous les états peuvent être régénérés avec le code dans `src/data_generation.py`

### Usage
```python
import pandas as pd

# Charger le dataset
df = pd.read_csv('data/processed/quantum_states_dataset.csv')

# Séparer features et target
X = df.drop(['state_id', 'is_valid'], axis=1)
y = df['is_valid']

# Ou utiliser la fonction du module
from src.data_generation import load_dataset
df = load_dataset()
```

### Fichiers

| Fichier | Description | Taille |
|---------|-------------|--------|
| `quantum_states_dataset.csv` | Dataset par défaut | ~1.5 MB |
| `quantum_states_dim4_n10000_seed42.csv` | Même dataset (nom descriptif) | ~1.5 MB |

### Notes

- Les features `c{i}_real` et `c{i}_imag` sont **indépendantes** (pas de corrélation forte)
- La feature `norm_squared` est **dérivée** des autres, mais utile pour le ML
- Le dataset est **mélangé** (shuffle=True) pour éviter tout biais d'ordre
- Aucune valeur manquante

### Versions

- **v1.0** (2024-11-12) : Dataset initial, 10k échantillons, dim=4

---

**Date de création :** 2025-11-12  
**Auteur :** MklZenin 
**Projet :** Quantum State Validator