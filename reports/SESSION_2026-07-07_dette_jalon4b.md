# Rapport de session — Purge de dette + Jalon 4b (dérive de calibration)

**Date :** 7 juillet 2026

## Partie 1 — Dette technique : purgée intégralement

| Item | Résolution |
|---|---|
| Q3 | `scale_features(method=...)` réel : standard / minmax / robust, validé, avec la justification du robust pour les queues lourdes du projet (effondrement LogReg du nb 08). Testé. |
| Q6 | `verify_normalization` : critère strictement absolu \|‖ψ‖²−1\| ≤ tol — l'ancien `np.allclose` gardait un `rtol=1e-5` implicite qui élargissait silencieusement le seuil annoncé. Verrouillé par un test qui refuse un écart de 2×10⁻⁵ à tol=10⁻⁶. |
| Q7 | Stratégie `basis` vectorisée (indexation avancée). Testé. |
| A4 (docstrings) | `generate_valid_states` retrouve sa docstring complète (dont l'argument mesure de Haar : gaussien isotrope + normalisation = uniforme sur la sphère). |
| A1/A2 (notebooks) | **04 et 06 rebranchés sur `src/paths` et ré-exécutés de bout en bout.** Flagrant délit de dérive : le 06 appelait des fonctions disparues de `preprocessing.py` (`create_logistic_regression`, `train_model`) — réécrit contre l'API actuelle. Le 04 aligné sur les paramètres du dataset versionné : **sa ré-exécution reproduit le CSV octet pour octet** (preuve : `git status` vide après exécution). |
| A3 | `main.tex` → `reports/`. Dossier `Fichiers annexes/` → `docs-annexes/`. |

Un commit mélangé (main.tex aspiré dans le commit notebooks) a été détecté et scindé avant push — le piège du staging partiel, attrapé cette fois.

## Partie 2 — Jalon 4b : la dérive de calibration

`add_calibration_drift` : gain multiplicatif g(t) = A·sin(2πt/T) + bruit de grenaille — le cycle thermique orbital d'un instrument embarqué. Notebook 10 exécuté, chiffres avant texte, dont une hypothèse intermédiaire réfutée (le RF avec temps ne bat PAS la recalibration).

| Candidat | Accuracy |
|---|---|
| Seuil fixe (champion des nb 08-09) | 0.9368 (contre 0.9724 sans dérive) |
| **Recalibration en ligne** (médiane glissante, zéro ML) | 0.9636 |
| RF 12 features + temps | 0.9388 |
| **GBM sur (norme, temps) seuls** | 0.9620 |
| **Hybride physique + ML** | **0.9668** |

Enseignements : (1) la non-stationnarité casse le seuil fixe — première défaite du champion ; (2) la réponse métier (recalibration) tient son rang ; (3) le ML échoue noyé dans les features mais apprend la carte de calibration avec la bonne représentation — troisième apparition de la leçon « représentation > algorithme » ; (4) **le meilleur système est hybride** : statistique physique en feature, ML pour les résidus — l'architecture des vrais FDIR.

## État

40 tests verts, 8 commits, arbre propre. Le jalon 4 a répondu à sa question centrale. Reste 4c (bruit coloré, états cibles, dims variables, abaques) et jalon 5 (API + interface pédagogique). Avancement : ~93 %.
