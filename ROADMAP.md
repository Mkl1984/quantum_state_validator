# Roadmap — Quantum State Validator

Roadmap vivante : mise à jour à chaque session de travail. Historique détaillé
dans `reports/` (audit du 2026-07-07 + rapports de session).

## Vision

Faire de QSV une référence pédagogique open source pour l'apprentissage de la
mécanique quantique et du ML rigoureux, structurée autour d'un fil conducteur
aérospatial : la qualification d'états quantiques sous budget de mesures fini,
telle qu'elle se pose dans les horloges GNSS, la navigation inertielle
quantique et la QKD par satellite.

## Jalons

### ✅ Jalon 1 — Théorie et génération de données (v0.1.0)
Modules de génération (3 stratégies valides, 4 invalides + cas extrêmes),
notebooks de théorie, dataset 10k.

### ✅ Jalon 2 — EDA et preprocessing (v0.2.0)
EDA avancée (Plotly 3D, simplex), split stratifié 60/20/20, scaling sans fuite.

### ✅ Jalon 3 — Évaluation honnête (v0.3.0)
- Notebook 07 archivé comme cas d'école de target leakage
- `src/features.py` : dichotomie features invariantes / sensibles à l'échelle
- Reformulation par bruit de mesure (notebook 08) : σ = 1/(2√N), biais 2dσ²,
  ROC par budget N, comparaison au test statistique optimal
- Garantie de frontière de classe (`norm_margin`), dataset régénéré (seed=42)
- 28 tests pytest, CI GitHub Actions, hygiène Git complète, docs open source

### 🔄 Jalon 4 — Là où le ML gagne sa place (v0.4.0) — EN COURS

**4a (fait, notebook 09)** :
- ✅ Bruit équicorrélé (`add_correlated_noise`) — **résultat négatif assumé** :
  le test à seuil résiste au mode commun (la somme sur 2d composantes filtre
  le mode commun). Valeur ingénierie : pas de modèle à embarquer pour ce régime.
- ✅ Diagnostic multiclasse de la cause (`create_multiclass_dataset`, stratégie
  `extreme` exposée) — RF à 90 %, mais dominé par la norme ; confusion
  scaling↔noise (40 %) démontrée comme **limite de Bayes par isotropie**.

**4b (à faire)** — briser l'isotropie et la stationnarité pour que le ML gagne :
- Dérives de calibration lentement variables (seuil fixe sous-optimal)
- Bruit coloré par composante (voies d'acquisition inégales)
- Causes corrélées à la direction : préparation défaillante d'états cibles
  *connus* (la référence directionnelle brise l'isotropie)
- Dimensions variables (d ≠ 4) et features indépendantes de d
- Abaques budget N ↔ taux d'erreur

Risque principal : garder chaque extension physiquement motivée.

### 🔮 Jalon 5 — Production et interface (v1.0.0)
- API de prédiction (FastAPI) + modèle exporté
- Interface pédagogique interactive (parcours guidés, quiz adaptatifs,
  validation étape par étape — voir la vision du projet)
- Rebranchement notebooks 04/06 sur `src/paths.py` + ré-exécution complète
- Documentation API générée

## Dette technique (traquée, non bloquante)

| Item | Origine audit | Effort |
|---|---|---|
| Paramètre `method` inutilisé dans `scale_features` | Q3 | 30 min |
| Sémantique de tolérance de `verify_normalization` (rtol implicite) | Q6 | 30 min |
| Vectorisation stratégie `basis` | Q7 | 15 min |
| `main.tex` → `reports/`, dossiers vides (`configs/`, `figures/`) | A3 | 15 min |
| Salvage des docstrings des `.pyi` supprimés vers les `.py` | A4 | 1-2 h |
| Notebooks 04/06 : chemins via `src/paths.py` + ré-exécution | A1 | 1-2 h |
| Renommage `Fichiers annexes/` (espaces + accents) | — | 15 min |
| Parquet si dataset > 10⁶ lignes | — | si besoin |
