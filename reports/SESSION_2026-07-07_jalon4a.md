# Rapport de session — Jalon 4a : où le ML gagne-t-il sa place ?

**Date :** 7 juillet 2026
**Objectif :** première tranche du jalon 4 (ROADMAP) — bruit corrélé + diagnostic multiclasse

## Méthode

Discipline renforcée après les leçons des phases précédentes : **les expériences ont été
exécutées avant d'écrire la moindre interprétation**. L'hypothèse initiale (« le RF bat le
seuil sous bruit corrélé ») a été réfutée par les données — le notebook le documente tel quel.

## Réalisé

| Livrable | Contenu |
|---|---|
| `add_correlated_noise` (features.py) | Bruit équicorrélé par mode commun : εᵢ = σ(√ρ·z + √(1−ρ)·wᵢ). Var inchangée, seule la structure varie (contrôle expérimental propre). Corrélation empirique vérifiée (0.895 pour ρ=0.9). |
| `create_multiclass_dataset` + stratégie `extreme` (data_generation.py) | Labels de cause (valid/scaling/noise/direct/extreme), garantie F2 conservée, reproductible. Analogie FDIR documentée. |
| Notebook 09 (exécuté) | Deux expériences, texte écrit après les chiffres. |
| Tests | +6 (structure de corrélation, ρ=0 ≡ i.i.d., validation, marge stratégie extreme, schéma multiclasse, reproductibilité). **Total : 34 verts.** ROADMAP/CHANGELOG à jour. |

## Résultats scientifiques

1. **Résultat négatif assumé** : le test à seuil résiste au bruit équicorrélé
   (0.9164→0.9104 entre ρ=0 et ρ=0.9 ; RF équivalent). Explication : le couplage mode
   commun↔signal passe par z·Σcᵢ, et la somme sur 2d=8 composantes filtre le mode commun.
   Valeur ingénierie : pour ce régime, un seuil calibré suffit — pas de modèle à qualifier.
2. **Diagnostic de cause : premier vrai métier du ML** (RF 0.899 là où un booléen ne dit
   rien), mais dominé par la norme (arbre 1D sur norme : 0.895). valid/extreme parfaits.
3. **Confusion scaling↔noise (40 %) = limite de Bayes**, démontrée par isotropie : les
   directions renormalisées des deux causes sont uniformes sur la sphère — seule la norme
   discrimine, et ses densités se chevauchent. Aucun modèle ne peut faire mieux.

## Décision de fond

Le jalon 4b devra **briser l'isotropie ou la stationnarité** (dérives temporelles de
calibration, bruit coloré par voie, états cibles connus) pour créer un régime où
l'apprentissage l'emporte légitimement. Inscrit dans la ROADMAP.

## Progression

34 tests verts, notebook 09 exécuté, avancement global ~88 % → **~91 %**.
Prochaines étapes : jalon 4b, ou dette technique (notebooks 04/06 → src/paths), au choix.
