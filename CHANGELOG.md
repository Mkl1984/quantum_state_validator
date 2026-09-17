# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ;
versionnement [SemVer](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté
- P0 : `qsv.tomography` — modèle de comptage exact (k_i ~ Poisson(N|c_i|²))
  remplaçant la simplification gaussienne σ = 1/(2√N), dernière limite
  scientifique déclarée du projet. Estimateur non biaisé (aucune correction
  2dσ² nécessaire), p-valeur de Poisson bilatérale exacte calculée par
  sommation en espace log (sans dépendance scipy), drapeau de budget
  N ≥ 4/marge². Notebook 15 (code-only) + section doc dans le 13 ; README,
  ROADMAP et section « limites » mis à jour ; 23 nouveaux tests (total 104).
  Résultat : l'abaque du notebook 12 survit au changement de modèle (accord
  à ~1 point), le terme 2dσ² est un artefact du modèle gaussien, et la
  non-identifiabilité de la norme à partir des seules fréquences est prouvée
- P0 : `qsv.density` — validation des matrices densité (hermiticité, trace
  unité, positivité) en modes exact et tomographique bruité ; seuils
  calibrés empiriquement (3600 essais) ; générateurs Ginibre + causes
  d'invalidité à violation garantie ; pont `from_state_vector` ; notebook 14
  (code-only) + section doc dans le 13 ; 13 nouveaux tests (total 81)
- P0 distribution : `qsv.adapters` (conversion duck-typée Qiskit/PennyLane/
  Cirq vers le cœur de décision — aucune dépendance framework), CLI
  `qsv validate` (fichiers .npy/.csv, codes de sortie 0/1/2 pour la CI,
  sortie JSON), artefacts PyPI construits et vérifiés, docs/RELEASING.md ;
  11 nouveaux tests (total 68)
- Phase finale : les 13 notebooks sont en anglais — 01-06 reconstruits
  sobres (convention « implémentation seule »), 07 archivé avec
  avertissement EN et sorties d'époque, 08-10 retraduits et ré-exécutés
  aux chiffres identiques ; CONTRIBUTING et data/README traduits
- Phase finale : README traduit intégralement en anglais (580 lignes,
  structure, résultats et 25 références préservés)
- Phase finale : l'intégralité du code (`src/qsv/` et `tests/`) est
  désormais en anglais — data_generation traduit avec preuve par hash MD5
  du dataset régénéré, 57 tests verts
- Phase finale : notebook 13 `13_project_report.ipynb` (EN, exécuté) —
  rapport de projet complet avec retour d'expérience des erreurs réelles
  à la première personne ; `qsv/features.py` traduit en anglais (57 tests
  valident la transcription)
- Jalon 5b : paquet installable `quantum-state-validator` (src-layout,
  pyproject) — logique de décision extraite dans `qsv/validators.py`,
  utilisable en bibliothèque (`from qsv import validate_state`) OU via le
  service HTTP (enveloppe mince), les deux combinables ; notebooks
  rebranchés sur le paquet et ré-exécutés ; CI sur `pip install -e .`
- Jalon 5a : API FastAPI (`src/api.py`, EN) servant les validateurs gagnants
  du jalon 4 (aucun modèle ML — décision d'architecture documentée) :
  `/validate` exact/bruité avec avertissement de budget, `/preparation-qa`
  deux canaux, docs OpenAPI ; 7 tests TestClient ; fastapi/httpx en CI

## [0.4.0] — 2026-07-07

Jalon 4 complet : « où le ML gagne-t-il sa place ? » — quatre régimes
étudiés, expériences avant interprétations, résultats négatifs assumés.

### Ajouté
- Jalon 4d : notebook 12 (exécuté, EN) — pipeline d-agnostique (validation
  plus facile quand d croît : concentration de la mesure), abaque
  N ↔ taux d'erreur (FPR ≤ 1 % ⇒ N ≈ 2000 à marge 0.05), tests paramétrés
- Jalon 4c : `src/preparation.py` (QA de préparation contre cibles connues,
  rédigé en anglais) et notebook 11 (exécuté, en anglais) : la référence
  directionnelle brise la limite d'isotropie du nb 09 ; statistique double
  (norme, fidélité) à 0.92 là où chacune plafonne à 0.667
- Jalon 4b : `add_calibration_drift` (dérive de gain sinusoïdale) et
  notebook 10 (exécuté) : effondrement du seuil fixe, recalibration en
  ligne, apprentissage de la carte de calibration, victoire de l'hybride
  physique+ML
- Q3 : `scale_features` accepte réellement standard/minmax/robust
- Jalon 4a : `add_correlated_noise` (bruit équicorrélé, mode commun),
  `create_multiclass_dataset` + stratégie `extreme` exposée
- Notebook `09_correlated_noise_multiclass.ipynb` (exécuté) : robustesse du
  test à seuil au mode commun (résultat négatif documenté), diagnostic de
  cause à 90 %, limite de Bayes scaling↔noise par argument d'isotropie
- 6 nouveaux tests (structure de corrélation, schéma multiclasse) — total 34

## [0.3.0] — 2026-07-07

Le jalon de l'honnêteté scientifique. Détails : `reports/AUDIT_2026-07-07.md`
et rapports de session Phase 0/1/2.

### Ajouté
- `src/features.py` : features invariantes vs sensibles à l'échelle,
  bruit de mesure tomographique σ = 1/(2√N)
- `src/paths.py` : chemins ancrés sur la racine du projet
- Notebook `08_measurement_noise.ipynb` (exécuté) : reformulation statistique,
  biais 2dσ², ROC par budget N, comparaison au test à seuil optimal
- 28 tests pytest (dont test mécanique anti-leakage et garantie F2)
- CI GitHub Actions : black + pytest sur Python 3.10 / 3.12
- LICENSE (MIT), CONTRIBUTING.md, ROADMAP.md, templates issues/PR
- `.gitattributes` (normalisation LF) ; dataset 10k versionné et reproductible

### Modifié (dette purgée)
- Q6 : `verify_normalization` à critère strictement absolu (rtol caché supprimé)
- Q7 : stratégie `basis` vectorisée
- A1/A2 : notebooks 04/06 rebranchés sur `src/paths`, ré-exécutés (le 04
  reproduit le dataset versionné octet pour octet)
- A3 : `main.tex` déplacé vers `reports/`

### Modifié
- `generate_invalid_states` : garantie de frontière de classe unique
  |‖ψ‖² − 1| ≥ `norm_margin` (défaut 0.05), toutes stratégies
- Dataset régénéré sous cette garantie (`seed=42`)
- `print()` → `logging` dans tous les modules (73 occurrences)
- README réécrit : résultats réels, applications aérospatiales détaillées,
  correction entropie de von Neumann → entropie de Shannon de la
  distribution de Born
- Codebase formatée black

### Corrigé
- **Target leakage du notebook 07** (norm_deviation dans les features
  d'entraînement) — notebook archivé comme cas d'école, avertissement en tête
- `.gitignore` : exception du dataset cassée, `tests/` ignoré par accident
- `joblib` manquant dans requirements.txt (dépendance transitive implicite)

### Supprimé
- Pseudo-stubs `.pyi` (copies annotées vouées à diverger)

## [0.2.0] — 2025-12-29

### Ajouté
- EDA avancée (notebook 05), preprocessing avec features quantiques
  (`src/preprocessing.py`), split stratifié 60/20/20, scaling sans fuite

## [0.1.0] — 2025-11-19

### Ajouté
- Structure du projet, environnement, notebooks de théorie quantique
- `src/data_generation.py` : 3 stratégies d'états valides, 4 d'invalides,
  cas extrêmes, création de dataset
