# Rapport de session — Phase 1 : correction scientifique

**Date :** 7 juillet 2026
**Objectif :** éliminer le target leakage du notebook 07 et reformuler le problème en vrai problème d'apprentissage statistique (audit §4, §7 Phase 1)

## Réalisé

| Livrable | Contenu |
|---|---|
| `src/features.py` | Séparation explicite **features invariantes d'échelle** (entropy_shannon, purity_normalized, participation_ratio, max_prob — aveugles à la norme par construction) vs **sensibles** (norm_squared, norm_deviation, entropy_raw, purity_raw — assumées comme fuites documentées). Modèle de bruit de mesure `add_measurement_noise` (σ = 1/(2√N), shot noise tomographique). Zéro `print`, docstrings avec formules disséquées et lecture en français. |
| `src/paths.py` | Chemins ancrés racine projet (correction A1 — fin des duplicatas de dataset créés par le cwd des notebooks). |
| `src/data_generation.py` | **Correction F2** : garantie centralisée \|‖ψ‖² − 1\| ≥ `norm_margin` (défaut 0.05) pour TOUT état invalide, toutes stratégies, propagée dans `mixed`. Boucle Python remplacée par du vectorisé. Docstring complète. |
| `data/README.md` | Règle de labellisation documentée (tableau + historique + note sur le dataset existant). |
| Notebook 07 | Cellule d'avertissement en tête : **cas d'école de target leakage archivé**, analyse des 3 niveaux de fuite, renvoi vers le 08. |
| **Notebook 08** (`08_measurement_noise.ipynb`) | La reformulation honnête, **exécuté de bout en bout** : bruit tomographique à budget N ∈ {50, 500, 5000}, biais de l'estimateur 2dσ² (mesuré 0.0398 vs théorie 0.0400), chevauchement des classes, LogReg vs RF vs **test à seuil bilatéral corrigé du biais**, ROC par budget. |
| Tests | `tests/test_features.py` + `tests/test_data_generation.py` + `conftest.py` : **25 tests, tous verts** — dont le test anti-leakage (invariance d'échelle vérifiée mécaniquement pour k ∈ {0.1, 0.5, 2, 37}) et la garantie F2 sur les 4 stratégies. |

## Résultats scientifiques du notebook 08

| N (mesures) | Test seuil (1 param) | LogReg | Random Forest |
|---|---|---|---|
| 50 | **0.9184** | 0.6016 | 0.9116 |
| 500 | **0.9704** | 0.6032 | 0.9692 |
| 5000 | **0.9928** | 0.6072 | 0.9920 |

Trois enseignements : (1) **le test statistique à 1 paramètre bat le Random Forest à tous les N** — toute l'information tient dans ‖ψ̂‖², comme la physique l'annonce ; (2) la performance est pilotée par le budget de mesures N, pas par l'algorithme — la vraie question d'ingénieur devient « quel N pour un taux de faux positifs contractuel ? » (dimensionnement type horloge GNSS / session QKD) ; (3) **découverte non anticipée** : la régression logistique s'effondre à ~60 % — géométrie de bande non linéairement séparable + standardisation écrasée par les queues lourdes (purity_raw ~10⁸). La représentation compte plus que l'algorithme.

## Décisions importantes

1. **Modèle de bruit gaussien additif** ĉ = c + ε, σ = 1/(2√N), assumé comme modèle simplifié documenté (le multinomial pur est norm-aveugle par construction — les statistiques de mesure ne voient jamais la norme, point souligné dans features.py).
2. Le label sous bruit reste la validité de l'état **sous-jacent** : c'est la définition correcte de la tâche de qualification.
3. Notebook 07 **archivé, pas supprimé** : l'erreur documentée est un actif pédagogique.
4. Le dataset 10k existant n'est **pas régénéré** (antérieur à F2) — noté dans data/README.md, prévu jalon 4.

## Problèmes rencontrés

- **Édition partielle de fichiers tronquée par le montage** (data_generation.py coupé à 637/718 lignes). Récupération : reconstruction depuis HEAD + patch scripté sur disque local + copie entière. Règle confirmée : sur ce montage, écritures de fichiers **complètes** uniquement.
- Interprétation initiale du markdown 08 invalidée par les résultats réels (LogReg ≠ « jeu égal ») : corrigée avant commit — on ne publie pas un texte contredit par ses propres sorties.

## Progression

| Poste | Avant Phase 1 | Après |
|---|---|---|
| Validité scientifique jalon 3 | invalide (leakage) | **reformulé et honnête (nb 08)** |
| Tests pytest | 0 | 25 verts |
| Frontière de classe | dépendante de la stratégie | unique, garantie, testée |
| Chemins cwd-dépendants (A1) | bug latent | src/paths.py |
| Avancement global | ~58 % | **~70 %** |

## Prochaines priorités (Phase 2 — audit §7)

1. `logging` dans data_generation/preprocessing (Q1), suppression des `.pyi` (A4), `joblib` dans requirements (Q5)
2. GitHub Action : lint (black) + pytest
3. Régénération du dataset post-F2 + notebooks 04/06 branchés sur `src/paths.py`
4. Puis Phase 3 : LICENSE, README réécrit avec les résultats du nb 08, ROADMAP.md
