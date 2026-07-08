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

**4b (fait, notebook 10)** :
- ✅ Dérive de calibration g(t) = A·sin(2πt/T) (`add_calibration_drift`) —
  le seuil fixe s'effondre (0.972→0.937) ; recalibration en ligne 0.964 ;
  GBM sur (norme, temps) seuls 0.962 (le ML apprend la carte de calibration
  SI la représentation est bonne) ; **hybride physique+ML vainqueur (0.967)**.
  Le jalon 4 a répondu à sa question : le ML gagne sa place en régime non
  stationnaire, marié à la structure physique.

**4c (reste)** :
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

**Purge du 2026-07-07** : Q3 (method réel : standard/minmax/robust), Q6
(tolérance strictement absolue), Q7 (basis vectorisé), A3 (main.tex →
reports/), A4 (docstrings restaurées), A1/A2 (notebooks 04/06 rebranchés
sur src/paths et ré-exécutés — le 04 reproduit le CSV octet pour octet),
dossier annexes renommé. Il reste :

| Item | Origine audit | Effort |
|---|---|---|
| Parquet si dataset > 10⁶ lignes | — | si besoin |
