# Quantum State Validator

**Classification d'États Quantiques par Machine Learning**

---

## Description

Ce projet implémente un pipeline complet de Machine Learning pour classifier automatiquement la validité d'états quantiques discrets, en combinant les principes de la mécanique quantique avec des techniques modernes d'apprentissage automatique.

### Problématique Physique

En mécanique quantique, un état pur |ψ⟩ est représenté par un vecteur dans un espace de Hilbert de dimension finie. Pour être physiquement valide, cet état doit satisfaire la **condition de normalisation** (Born's rule) :

$$\sum_{i=1}^{n} |c_i|^2 = 1$$

où les $c_i \in \mathbb{C}$ sont les amplitudes de probabilité et $|c_i|^2$ représente la probabilité de mesurer l'état dans la base $|i\rangle$.

### Défi Machine Learning

**Objectif** : Entraîner un classifieur binaire capable de distinguer états valides (normalisés) et invalides (non-normalisés) **sans accès direct à la norme**, forçant le modèle à apprendre les patterns sous-jacents via des features d'ingénierie quantique.

**Originalité** : Exclusion délibérée de `norm_squared` du training pour éviter la solution triviale et simuler un scénario où la validation directe est coûteuse ou impossible.

---

## Objectifs Pédagogiques

### Compétences ML Développées

- Génération de données synthétiques avec contraintes physiques (Dirichlet, perturbations contrôlées)
- Feature engineering spécialisé (entropy de von Neumann, purity quantique)
- Pipeline ML complet : preprocessing, training, validation, interprétation
- Évaluation rigoureuse : matrice de confusion, ROC curves, learning curves
- Optimisation d'hyperparamètres (Bayesian optimization avec Optuna)
- Visualisation avancée : Plotly 3D interactif, analyses géométriques
- Bonnes pratiques : versioning Git, structure modulaire, documentation

### Concepts Physiques Appliqués

- Born's rule et normalisation quantique
- Espaces de Hilbert discrets et simplex de probabilités
- Entropie de von Neumann et mesure de pureté
- Génération d'états aléatoires sur le simplex (distribution de Dirichlet)

---

## Architecture du Projet

```
quantum_state_validator/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── quantum_states_10k.csv
│
├── notebooks/
│   ├── 01_quantum_theory.ipynb
│   ├── 02_test_data_generation.ipynb
│   ├── 03_test_invalid_states.ipynb
│   ├── 04_create_dataset.ipynb
│   ├── 05_eda_advanced.ipynb
│   ├── 06_test_preprocessing.ipynb
│   └── 07_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── data_generation.pyi
│   ├── preprocessing.py
│   └── preprocessing.pyi
│
├── models/
├── results/
├── docs/
│   └── 07_GUIDE_THEORIQUE.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### Prérequis

- Python 3.10+
- pip ou conda
- 4 GB RAM minimum (8 GB recommandé)

### Étapes

#### 1. Cloner le Dépôt

```bash
git clone https://github.com/USERNAME/quantum_state_validator.git
cd quantum_state_validator
```

#### 2. Créer un Environnement Virtuel

**Avec venv (Windows)** :

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Avec venv (macOS/Linux)** :

```bash
python -m venv venv
source venv/bin/activate
```

**Avec conda** :

```bash
conda create -n qsv_env python=3.11 -y
conda activate qsv_env
```

#### 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales** :
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.14.0
- jupyter >= 1.0.0

#### 4. Vérifier l'Installation

```bash
python -c "from src.data_generation import create_dataset; print('Installation réussie')"
```

---

## Guide d'Utilisation

### Génération de Données

```python
from src.data_generation import create_dataset, save_dataset

# Génération du dataset
df = create_dataset(
    n_valid=5000,
    n_invalid=5000,
    dim=4,
    valid_strategy="random",
    invalid_strategy="mixed",
    seed=42
)

# Sauvegarde
save_dataset(df, "data/processed/quantum_states_10k.csv")
```

**Stratégies de génération** :

- **États valides** :
  - `random` : Distribution uniforme sur le simplex (Dirichlet)
  - `dirichlet` : Contrôle via paramètres α
  - `basis` : États de base

- **États invalides** :
  - `scaling` : Multiplication par facteur ≠ 1
  - `noise` : Ajout de bruit gaussien
  - `direct` : Génération aléatoire sans contrainte
  - `mixed` : Combinaison des stratégies

### Preprocessing et Feature Engineering

```python
from src.preprocessing import QuantumPreprocessor

preprocessor = QuantumPreprocessor()
preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

**Features calculées** :
- `entropy` : Entropie de von Neumann $S = -\sum_i p_i \log_2(p_i)$
- `purity` : Pureté quantique $P = \sum_i p_i^2$
- `norm_deviation` : Écart à la normalisation $|\sum_i |c_i|^2 - 1|$

### Entraînement

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modèle
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)
```

### Évaluation

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

---

## Résultats

### Dataset

| Métrique | Valeur |
|----------|--------|
| Taille totale | 10,000 échantillons |
| États valides | 5,000 (50%) |
| États invalides | 5,000 (50%) |
| Dimensions | 4 (4 qubits) |
| Features brutes | 8 (composantes réelles/imaginaires) |
| Features engineered | 3 (entropy, purity, norm_deviation) |
| Split | 60% train / 20% validation / 20% test |

### Performances Modèle

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| Random Forest | À compléter | À compléter | À compléter | À compléter | À compléter |

---

## Fondements Théoriques

### Mécanique Quantique

#### Espace de Hilbert et États Purs

Un état quantique pur en dimension $n$ est un vecteur $|\psi\rangle$ de l'espace de Hilbert $\mathcal{H} \cong \mathbb{C}^n$ :

$$|\psi\rangle = \sum_{i=0}^{n-1} c_i |i\rangle$$

où $\{|i\rangle\}_{i=0}^{n-1}$ est une base orthonormée.

#### Born's Rule (Règle de Born)

La probabilité de mesurer l'état $|i\rangle$ est donnée par :

$$P(i) = |c_i|^2 = |\langle i | \psi \rangle|^2$$

#### Condition de Normalisation

Pour que les probabilités soient cohérentes (somme = 1) :

$$\langle \psi | \psi \rangle = \sum_{i=0}^{n-1} |c_i|^2 = 1$$

Cette condition définit la **sphère unité** dans $\mathbb{C}^n$.

#### Le Simplex de Probabilités

En considérant uniquement les probabilités $p_i = |c_i|^2$, les états valides forment un **simplex** :

$$\Delta^{n-1} = \left\{ (p_0, \ldots, p_{n-1}) \in \mathbb{R}^n : p_i \geq 0, \sum_i p_i = 1 \right\}$$

**Propriétés géométriques** :
- Dimension : $n-1$
- Frontière : Hyperplan $\sum_i p_i = 1$
- Exemple 3D (n=4) : Tétraèdre régulier

### Features Quantiques

#### 1. Entropie de von Neumann

$$S(\psi) = -\sum_{i=0}^{n-1} p_i \log_2(p_i)$$

**Interprétation** :
- $S = 0$ : État pur concentré
- $S = \log_2(n)$ : État maximalement mixte
- Mesure de l'incertitude/désordre de l'état

#### 2. Pureté Quantique

$$P(\psi) = \sum_{i=0}^{n-1} p_i^2$$

**Propriétés** :
- $P = 1$ : État pur
- $P = 1/n$ : État maximalement mixte
- Relation avec entropie : $P$ et $S$ sont inversement corrélés
- Lien avec la trace : $P = \text{Tr}(\rho^2)$ où $\rho = |\psi\rangle\langle\psi|$

#### 3. Déviation de la Norme

$$D(\psi) = \left| \sum_{i=0}^{n-1} |c_i|^2 - 1 \right|$$

**Interprétation** :
- $D = 0$ : État valide
- $D > 0$ : État invalide

**Note** : Cette feature est calculée mais **exclue du training** pour éviter la solution triviale.

### Théorèmes Utilisés

**Théorème (Riesz-Fischer)** : L'espace de Hilbert $\mathcal{H}$ est complet pour la norme induite par le produit scalaire.

**Théorème (Spectral)** : Pour un opérateur hermitien $A$, il existe une base orthonormée de vecteurs propres.

**Lemme (Cauchy-Schwarz)** : Pour tout $|\psi\rangle, |\phi\rangle \in \mathcal{H}$ :

$$|\langle \psi | \phi \rangle|^2 \leq \langle \psi | \psi \rangle \cdot \langle \phi | \phi \rangle$$

---

## Références

### Mécanique Quantique

1. **Cohen-Tannoudji, C., Diu, B., & Laloë, F.** (2019). *Mécanique quantique* (Tomes 1 & 2). EDP Sciences.
   - Référence francophone complète, traitement rigoureux des espaces de Hilbert

2. **Basdevant, J.-L., & Dalibard, J.** (2002). *Mécanique quantique*. Éditions de l'École Polytechnique.
   - Approche pédagogique française, excellente pour les fondations

3. **CNRS - Introduction à la mécanique quantique** (2020). *Partie 1 : Postulats et formalisme*. Cours en ligne.
   - Accessible : https://www.cnrs.fr/fr/

4. **CNRS - Introduction à la mécanique quantique** (2020). *Partie 2 : Mesure et évolution*. Cours en ligne.
   - Complément de la partie 1

5. **Nielsen, M. A., & Chuang, I. L.** (2010). *Quantum Computation and Quantum Information* (10th Anniversary Edition). Cambridge University Press.
   - Référence internationale, chapitres 2-3 sur postulats et mesures

6. **Sakurai, J. J., & Napolitano, J.** (2017). *Modern Quantum Mechanics* (2nd Edition). Cambridge University Press.
   - Chapitre 1 : Espaces de Hilbert et états quantiques

7. **Griffiths, D. J., & Schroeter, D. F.** (2018). *Introduction to Quantum Mechanics* (3rd Edition). Cambridge University Press.
   - Excellent pour l'intuition physique

8. **Le Bellac, M.** (2013). *Physique quantique* (2e édition). EDP Sciences.
   - Approche moderne avec applications actuelles

### Machine Learning

9. **Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition). O'Reilly.
   - Chapitre 3 : Classification
   - Chapitre 6 : Arbres de décision et Random Forests

10. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
    - Chapitre 1.5 : Théorie de la décision
    - Approche mathématique rigoureuse

11. **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2021). *An Introduction to Statistical Learning* (2nd Edition). Springer.
    - Chapitre 4 : Classification
    - Très accessible avec exemples R/Python

12. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd Edition). Springer.
    - Référence avancée pour théorie statistique

### Articles Scientifiques

13. **Fawcett, T.** (2006). "An introduction to ROC analysis". *Pattern Recognition Letters*, 27(8), 861-874.
    - Référence complète sur courbes ROC et AUC

14. **Breiman, L.** (2001). "Random Forests". *Machine Learning*, 45(1), 5-32.
    - Article original sur Random Forests

15. **Lundberg, S. M., & Lee, S. I.** (2017). "A Unified Approach to Interpreting Model Predictions". *Advances in Neural Information Processing Systems*, 30.
    - Fondements théoriques de SHAP

16. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier". *Proceedings of KDD*, 1135-1144.
    - Introduction à LIME (Local Interpretable Model-agnostic Explanations)

### Quantum Machine Learning

17. **Schuld, M., & Petruccione, F.** (2018). *Supervised Learning with Quantum Computers*. Springer.
    - Pont entre ML classique et quantique

18. **Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., & Lloyd, S.** (2017). "Quantum machine learning". *Nature*, 549(7671), 195-202.
    - Review sur état de l'art QML

### Documentation Technique

19. **Scikit-learn Documentation** : https://scikit-learn.org/stable/
    - Module metrics : https://scikit-learn.org/stable/modules/model_evaluation.html
    - Guide utilisateur complet

20. **Plotly Documentation** : https://plotly.com/python/
    - Visualisations 3D : https://plotly.com/python/3d-charts/

21. **NumPy Documentation** : https://numpy.org/doc/stable/
    - Référence pour calculs numériques

### Ressources en Ligne

22. **Quantum Country** : https://quantum.country/
    - Introduction interactive à la mécanique quantique

23. **3Blue1Brown** : https://www.youtube.com/c/3blue1brown
    - Série "Essence of Linear Algebra"
    - Visualisations exceptionnelles pour espaces vectoriels

24. **StatQuest (Josh Starmer)** : https://www.youtube.com/c/joshstarmer
    - Série sur Random Forests et métriques ML
    - Pédagogie claire et rigoureuse

25. **Cours Collège de France - Alain Aspect** : https://www.college-de-france.fr/
    - Cours "Physique quantique" (2020-2021)
    - Niveau avancé mais accessible

---

## Roadmap du Projet

### Jalon 0 : Configuration
- Environnement virtuel Python
- Structure de projet professionnelle
- Git initialisé et configuré
- Dépendances installées

### Jalon 1 : Théorie et Données
- Compréhension théorique (Born rule, normalisation)
- Module `data_generation.py` avec type stubs
- Stratégies de génération (Dirichlet, perturbations)
- Tests de validation
- Dataset final 10,000 échantillons

### Jalon 2 : EDA et Preprocessing
- Analyse exploratoire avancée (Plotly 3D)
- Feature engineering quantique (entropy, purity)
- Classe `QuantumPreprocessor`
- Split stratifié train/val/test (60/20/20)
- Tests de preprocessing
- Visualisations géométriques du simplex

### Jalon 3 : Évaluation et Diagnostic
- Entraînement Random Forest baseline
- Métriques de classification complètes
- Matrice de confusion détaillée
- Courbe ROC et optimisation seuil
- Analyse géométrique des erreurs
- Learning curves et diagnostic overfitting
- Guide théorique complet

### Jalon 4 : Optimisation
- Feature importance et sélection
- Optimisation hyperparamètres (Optuna/GridSearch)
- Modèles alternatifs (Gradient Boosting, SVM)
- Interprétabilité (SHAP values, LIME)
- Cross-validation robuste
- Calibration des probabilités

### Jalon 5 : Production et Documentation
- Export modèle production
- API de prédiction (Flask/FastAPI)
- Tests unitaires complets
- Rapport technique complet
- Présentation professionnelle
- Documentation API

---

## Technologies

### Langages et Frameworks

| Technologie | Version | Usage |
|-------------|---------|-------|
| Python | 3.10+ | Langage principal |
| NumPy | 1.24+ | Calculs numériques |
| Pandas | 2.0+ | Manipulation de données |
| Scikit-learn | 1.3+ | Algorithmes ML |
| Matplotlib | 3.7+ | Visualisations statiques |
| Seaborn | 0.12+ | Visualisations statistiques |
| Plotly | 5.14+ | Visualisations 3D interactives |
| Jupyter | 1.0+ | Notebooks interactifs |

### Outils de Développement

- IDE : VS Code avec extensions Python, Jupyter
- Version Control : Git + GitHub
- Environment : venv ou conda
- Type Checking : mypy (via `.pyi` stubs)

---

## Contribution

### Branching Strategy

```bash
main              # Code stable
dev               # Intégration continue
feature/nom       # Nouvelles fonctionnalités
fix/nom           # Corrections
```

### Convention de Commits

Format : `<type>(<scope>): <sujet>`

Types :
- `feat` : Nouvelle fonctionnalité
- `fix` : Correction de bug
- `docs` : Documentation
- `style` : Formatage
- `refactor` : Refactorisation
- `test` : Ajout de tests
- `chore` : Maintenance

Exemples :
```bash
git commit -m "feat(jalon3): Ajouter courbes ROC avec optimisation seuil"
git commit -m "fix(preprocessing): Corriger calcul entropy pour p=0"
git commit -m "docs: Mettre à jour README avec résultats"
```

---

## Contact

**Auteur** : Mklzenin  
Étudiant MSc Ai & ML/DL

**GitHub** : https://github.com/Mkl1984/quantum_state_validator

Pour questions ou suggestions :
- Issues GitHub : https://github.com/Mkl1984/quantum_state_validator/issues
- Discussions : https://github.com/Mkl1984/quantum_state_validator/discussions

---

## Licence

Ce projet est développé dans un cadre pédagogique.

**MIT License**

Copyright (c) 2024 Mandem

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Citation

Si vous utilisez ce code dans votre travail, merci de citer :

```bibtex
@misc{quantum_state_validator_2024,
  author = {Mklzenin},
  title = {Quantum State Validator: ML Classification of Quantum States},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Mkl1984/quantum_state_validator}}
}
```

---

## Perspectives

### Extensions Possibles

- Dimensions supérieures (n > 4 qubits)
- États mixtes (matrices densité)
- Deep Learning (MLP, CNN)
- Transfer Learning
- Quantum ML (circuits variationnels)
- Données expérimentales réelles
- Classification multi-classes (pur, mixte, intriqué)

### Applications

- Contrôle qualité dans ordinateurs quantiques
- Validation d'états en simulations
- Outil de recherche pour physiciens

---

Dernière mise à jour : Décembre 2025  
Version : 0.3.0
