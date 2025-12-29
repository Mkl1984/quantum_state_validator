# Quantum State Validator

Projet d'apprentissage : Classification d'états quantiques par Machine Learning

---

## 📋 Description

Ce projet implémente un pipeline complet de Machine Learning pour vérifier la validité d'états quantiques discrets.

**Objectif physique :**  
Un état quantique |ψ⟩ = (c₁, c₂, ..., cₙ) est valide si et seulement si :
```
Σᵢ |cᵢ|² = 1
```

**Objectif ML :**  
Entraîner un classifieur binaire capable de distinguer états valides/invalides à partir de leurs composantes.

---

## 🎯 Compétences développées

- Génération de données synthétiques avec contraintes physiques
- Pipeline ML complet : preprocessing, training, validation, interprétation
- Visualisation 2D/3D d'états quantiques
- Versioning avec Git
- Structure de projet professionnelle

---

## 📁 Structure du projet
```
quantum_state_validator/
├── data/               # Données brutes et traitées
├── notebooks/          # Analyses interactives
├── src/                # Code source modulaire
├── models/             # Modèles entraînés
├── figures/            # Graphiques exportés
├── reports/            # Rapports finaux
└── README.md           # Ce fichier
```

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- pip ou conda

### Étapes

1. Clone le dépôt (ou télécharge le ZIP)
```bash
git clone [URL_DU_REPO]
cd quantum_state_validator
```

2. Crée un environnement virtuel

**Avec venv :**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # macOS/Linux
```

**Avec conda :**
```bash
conda create -n qsv_env python=3.11 -y
conda activate qsv_env
```

3. Installe les dépendances
```bash
pip install -r requirements.txt
```

4. Vérifie l'installation
```bash
jupyter notebook notebooks/00_smoke_test.ipynb
```

---

## 📊 Usage

*(À compléter au fur et à mesure)*

### Génération des données
```python
# TODO
```

### Entraînement
```python
# TODO
```

### Évaluation
```python
# TODO
```

---

## 📈 Résultats

*(À compléter après le jalon 2)*

| Modèle | Accuracy | F1-Score | Remarques |
|--------|----------|----------|-----------|
| Logistic Regression | TBD | TBD | Baseline |
| Random Forest | TBD | TBD | Modèle final |

---

## 🔬 Concepts physiques

### Normalisation d'un état quantique

Un état quantique pur en dimension finie est représenté par un vecteur complexe :
```
|ψ⟩ = Σᵢ cᵢ|i⟩
```

où :
- `cᵢ ∈ ℂ` sont les amplitudes de probabilité
- `|cᵢ|²` = probabilité de mesurer l'état dans la base |i⟩
- Condition de normalisation : `Σᵢ |cᵢ|² = 1`

Cette condition garantit que la somme des probabilités = 1 (cohérence probabiliste).

---

## 📚 Références

- Nielsen & Chuang, *Quantum Computation and Quantum Information*
- Scikit-learn documentation
- Cours de Mécanique Quantique, MSc IA/ML

---

## ✅ Checklist de progression

- [x] Setup environnement
- [x] Structure projet
- [ ] Génération données
- [ ] Visualisation exploratoire
- [ ] Baseline ML
- [ ] Optimisation
- [ ] Rapport final

---

## 👤 Auteur

**[Ton Nom]**  
MSc Intelligence Artificielle & Machine Learning  
*Projet d'apprentissage - Portfolio Data Scientist*

---

## 📜 Licence

Ce projet est à usage éducatif.