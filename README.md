# Quantum State Validator

**Classification d'États Quantiques par Machine Learning**

[![CI](https://github.com/Mkl1984/quantum_state_validator/actions/workflows/ci.yml/badge.svg)](https://github.com/Mkl1984/quantum_state_validator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Description

Ce projet implémente un pipeline complet de Machine Learning pour classifier automatiquement la validité d'états quantiques discrets, en combinant les principes de la mécanique quantique avec des techniques d'apprentissage automatique.

### Problématique Physique

En mécanique quantique, un état pur |ψ⟩ est représenté par un vecteur dans un espace de Hilbert de dimension finie. Pour être physiquement valide, cet état doit satisfaire la **condition de normalisation** (Born's rule) :

$$\sum_{i=1}^{n} |c_i|^2 = 1$$

où les $c_i \in \mathbb{C}$ sont les amplitudes de probabilité et $|c_i|^2$ représente la probabilité de mesurer l'état dans la base $|i\rangle$.

### Défi Machine Learning

Sur amplitudes **exactes**, la validité est une fonction déterministe des features — ‖ψ‖² se calcule, il n'y a rien à apprendre. Le problème est alors soit trivial (avec une feature sensible à l'échelle), soit impossible (avec uniquement des features invariantes d'échelle). Le notebook 07, conservé volontairement comme **cas d'école de target leakage**, documente cette impasse : exclure `norm_squared` tout en gardant `norm_deviation = |norm_squared − 1|` produisait 100 % d'accuracy... en entraînant le modèle sur la définition du label.

**La formulation honnête** (notebook 08) : les amplitudes sont observées à travers un **bruit de tomographie à budget de mesures fini** N, soit ĉᵢ = cᵢ + ε avec σ = 1/(2√N) — la loi du bruit de grenaille (*shot noise*). Près de la frontière ‖ψ‖² = 1, les classes se chevauchent réellement : décider devient un problème statistique légitime, avec de vrais compromis faux positifs / faux négatifs pilotés par le budget N. C'est le problème opérationnel exact d'un système de **qualification d'états quantiques embarqué** (voir Applications aérospatiales).

---

## Résultats Clés

Notebook 08 — accuracy de test selon le budget de mesures N (10 000 états, dim = 4, seed = 42) :

| N (mesures) | Test à seuil corrigé du biais (1 paramètre) | Régression logistique | Random Forest (100 arbres) |
|---|---|---|---|
| 50 | **0.913** | 0.603 | 0.906 |
| 500 | **0.972** | 0.602 | 0.974 |
| 5 000 | **0.997** | 0.606 | 0.998 |

Trois enseignements :

1. **Le test statistique à un paramètre fait jeu égal avec le Random Forest** : toute l'information de validité tient dans la norme estimée ‖ψ̂‖², comme la physique l'annonce. Comprendre la structure du problème vaut mieux qu'empiler des modèles.
2. **La performance est pilotée par la physique** (le budget N), pas par l'algorithme. La question d'ingénierie devient : quel N minimal pour un taux de faux positifs contractuel ?
3. **La régression logistique s'effondre (~60 %)** : la classe valide vit dans une *bande* de norme, géométrie non séparable linéairement, aggravée par les features à queues lourdes. La représentation compte plus que l'algorithme.

Le biais de l'estimateur de norme mesuré (E[‖ψ̂‖²] − ‖ψ‖² = 0.0398 à N = 50) coïncide avec la prédiction théorique 2dσ² = 0.0400 — le modèle de bruit se comporte comme la théorie l'exige.

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
├── .github/workflows/ci.yml     # CI : black + pytest (Python 3.10 / 3.12)
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── quantum_states_10000.csv   # 10 000 états, dim=4, seed=42 (reproductible)
│   └── README.md                # Schéma + règle de labellisation (frontière F2)
├── notebooks/
│   ├── 01_quantum_theory.ipynb        # Théorie : normalisation, règle de Born
│   ├── 02_test_data_generation.ipynb  # Génération d'états valides (3 stratégies)
│   ├── 03_test_invalid_states.ipynb   # Génération d'états invalides (4 stratégies)
│   ├── 04__create_dataset.ipynb       # Construction du dataset
│   ├── 05_eda_advanced.ipynb          # EDA avancée (Plotly 3D, simplex)
│   ├── 06_test_preprocessing.ipynb    # Pipeline de preprocessing
│   ├── 07_model_evaluation.ipynb      # ARCHIVÉ : cas d'école de target leakage
│   └── 08_measurement_noise.ipynb     # Formulation honnête : bruit de mesure
├── src/
│   ├── data_generation.py       # États valides/invalides, garantie de frontière F2
│   ├── features.py              # Features invariantes vs sensibles + bruit tomographique
│   ├── paths.py                 # Chemins ancrés sur la racine du projet
│   └── preprocessing.py         # Split stratifié 60/20/20 + scaling sans fuite
├── tests/                       # 28 tests pytest (anti-leakage, garantie F2, splits)
├── reports/                     # Audit complet + rapports de session
├── CHANGELOG.md · CONTRIBUTING.md · LICENSE · ROADMAP.md
└── requirements.txt
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
git clone https://github.com/Mkl1984/quantum_state_validator.git
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
pytest tests/ -q        # 28 tests doivent passer
```

La même vérification tourne en intégration continue (GitHub Actions) à chaque push : formatage `black --check` puis suite `pytest` sur Python 3.10 et 3.12.

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

Le module `src/features.py` sépare les features en deux familles, et cette séparation est **le cœur méthodologique du projet** :

- **Invariantes d'échelle** (calculées sur p̃ᵢ = pᵢ/Σpⱼ) : insensibles à c → k·c, donc structurellement aveugles à la validité. Un classifieur qui n'utilise qu'elles plafonne à ~50 % — c'est un théorème, pas un échec.
- **Sensibles à l'échelle** (calculées sur les pᵢ bruts) : elles encodent la norme, donc le label. Sur données exactes ce sont des fuites ; sur données bruitées (notebook 08) ce sont des estimateurs légitimes.

Cette propriété est verrouillée par un test mécanique (`tests/test_features.py::test_invariant_features_are_scale_invariant`).

#### 1. Entropie de Shannon de la distribution de Born

$$H(\tilde{p}) = -\sum_{i=0}^{n-1} \tilde{p}_i \ln(\tilde{p}_i)$$

**Interprétation** :
- $H = 0$ : état de base pur ; $H = \ln(n)$ : superposition uniforme
- Mesure l'étalement de l'état sur la base de mesure

**Précision physique** : pour un état *pur*, l'entropie de von Neumann $S(\rho) = -\mathrm{Tr}(\rho \ln \rho)$ vaut exactement 0. La quantité utile ici est l'entropie de Shannon de la distribution de mesure $\tilde{p}_i$ (règle de Born) dans la base de calcul — d'où le nom `entropy_shannon` dans le code (en nats : logarithme naturel ; conversion en bits : facteur $1/\ln 2$).

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

**Attention** : sur amplitudes exactes, cette quantité *définit* le label — l'utiliser en feature est du target leakage (leçon du notebook 07). Sur amplitudes bruitées (notebook 08), sa version estimée devient la statistique de décision légitime, à corriger du biais $2d\sigma^2$.

### Théorèmes Utilisés

**Théorème (Riesz-Fischer)** : L'espace de Hilbert $\mathcal{H}$ est complet pour la norme induite par le produit scalaire.

**Théorème (Spectral)** : Pour un opérateur hermitien $A$, il existe une base orthonormée de vecteurs propres.

**Lemme (Cauchy-Schwarz)** : Pour tout $|\psi\rangle, |\phi\rangle \in \mathcal{H}$ :

$$|\langle \psi | \phi \rangle|^2 \leq \langle \psi | \psi \rangle \cdot \langle \phi | \phi \rangle$$

---

## Applications — Aérospatial et Aéronautique

Ce projet est un **modèle réduit pédagogique du sous-système de qualification d'états** présent dans tout capteur quantique embarqué. Le fil conducteur : un instrument quantique en vol ne connaît jamais ses amplitudes exactes — il doit décider de la conformité d'un état à partir de **statistiques de mesure finies**, sous contrainte de ressources. C'est exactement la structure du notebook 08.

### 1. Horloges atomiques de navigation par satellite (GNSS)

Les constellations GPS (horloges rubidium RAFS) et Galileo (masers à hydrogène passifs PHM + RAFS) reposent sur l'interrogation d'états atomiques. La stabilité de fréquence (variance d'Allan) s'améliore en 1/√N avec le nombre d'interrogations — la même loi que notre σ = 1/(2√N). Le dilemme du notebook 08 (budget N ↔ taux de faux positifs) est celui du dimensionnement du temps d'intégration d'une horloge pour tenir une spécification de stabilité, dont dépend directement l'erreur de positionnement au sol (1 ns d'erreur d'horloge ≈ 30 cm d'erreur de pseudo-distance).

### 2. Navigation inertielle quantique (interférométrie atomique)

Les gyromètres et accéléromètres à atomes froids (travaux ONERA — gravimètre GIRAFE marinisé, projets de centrales inertielles quantiques pour la navigation sans GNSS) exigent la préparation d'états atomiques conformes avant chaque cycle d'interférométrie. En environnement vibratoire (porteur aéronautique), la qualification de l'état préparé à partir de mesures partielles est une étape critique : un état mal préparé accepté = biais de navigation ; un état correct rejeté = perte de cadence de mesure. C'est le compromis FP/FN de notre courbe ROC.

### 3. Distribution quantique de clés par satellite (QKD)

Le satellite Micius (2016, liaison QKD sol-espace de 1 200 km) et les programmes européens EuroQCI/Eagle-1 qualifient des états photoniques (polarisation, intrication) à partir d'échantillons finis : le taux d'erreur quantique (QBER) estimé sur un sous-ensemble de mesures décide si la clé est sûre ou jetée. Budget de photons ↔ notre budget N ; seuil de QBER ↔ notre seuil sur ‖ψ̂‖².

### 4. Contrôle qualité des calculateurs quantiques pour l'optimisation aérospatiale

Les cas d'usage étudiés par les industriels (optimisation de trajectoires, ordonnancement de charges utiles, logistique de flotte — Airbus a lancé un Quantum Computing Challenge dédié) supposent des registres de qubits correctement préparés. La **tomographie de vérification** entre deux campagnes de calcul est un problème de validation d'états sous budget de mesures — dimensionner ce budget, c'est notre question centrale.

### 5. Assurance qualité des simulations numériques quantiques

Le calcul de structures électroniques pour matériaux aérospatiaux (alliages, catalyseurs, batteries) produit des vecteurs d'état numériques dont la norme dérive avec les erreurs d'arrondi et les schémas d'intégration. Un validateur de type QSV joue le rôle de **garde-fou de pipeline** (data QA) : détecter une dérive de normalisation avant qu'elle ne contamine les observables calculées.

### 6. Magnétométrie quantique embarquée

Les magnétomètres à pompage optique et à centres NV pour la détection d'anomalies magnétiques (navigation par corrélation de cartes, prospection aéroportée) reposent sur des états de spin dont la qualité de préparation conditionne la sensibilité. Diagnostic embarqué = décision statistique sur mesures finies, à cadence imposée par la dynamique du porteur.

**Lecture transverse** : dans les six cas, la grandeur d'ingénierie n'est pas « l'accuracy du modèle » mais le couple (budget de mesures N, taux d'erreur toléré) — et la leçon du projet est qu'un test statistique bien construit sur la bonne statistique suffit souvent, le ML n'apportant de valeur qu'en présence de bruit structuré (corrélations, dérives de calibration : voir ROADMAP jalon 4).

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

Roadmap vivante et détaillée : voir [ROADMAP.md](ROADMAP.md). Synthèse :

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

### Jalon 3 : Évaluation Honnête — TERMINÉ (v0.3.0)
- Notebook 07 : première évaluation, invalidée par target leakage — **archivée comme cas d'école**
- Reformulation avec bruit de mesure (notebook 08) : ROC par budget N, biais 2dσ², comparaison au test statistique optimal
- Garantie de frontière de classe (norm_margin) + dataset régénéré reproductible
- 28 tests pytest (anti-leakage mécanique inclus) + CI GitHub Actions

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
- Version Control : Git + GitHub (Conventional Commits, tags de release)
- Environment : venv ou conda
- Qualité : black (formatage), pytest (28 tests), GitHub Actions (CI sur 3.10/3.12)
- Annotations de type directement dans les modules (les stubs `.pyi` ont été retirés)

---

## Contribution

Guide complet : [CONTRIBUTING.md](CONTRIBUTING.md). L'essentiel :

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
Étudiant ingénieur Aérospatial (spécialité AI and cybersécurity)

**GitHub** : https://github.com/Mkl1984/quantum_state_validator

Pour questions ou suggestions :
- Issues GitHub : https://github.com/Mkl1984/quantum_state_validator/issues
- Discussions : https://github.com/Mkl1984/quantum_state_validator/discussions

---

## Licence

Fruit d'une curiosité personnelle pour l'univers quantique, ce projet a évolué vers un véritable support d'apprentissage. Son objectif est d'offrir une interface claire et intuitive pour s'initier aux principes de base de la mécanique quantique, sans sacrifier la précision du propos.

**MIT License**

Copyright (c) 2024 Mandem

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Citation

Si vous utilisez ce code dans votre travail, merci de citer :

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

### Extensions Possibles

- Dimensions supérieures (n > 4 qubits)
- États mixtes (matrices densité)
- Deep Learning (MLP, CNN)
- Transfer Learning
- Quantum ML (circuits variationnels)
- Données expérimentales réelles
- Classification multi-classes (pur, mixte, intriqué)

### Applications

Voir la section détaillée [Applications — Aérospatial et Aéronautique](#applications--aérospatial-et-aéronautique).

---

Dernière mise à jour : Juillet 2026  
Version : 0.3.0 — voir [CHANGELOG.md](CHANGELOG.md)
