# Rapport de session — Phase 3 : ouverture open source et release v0.3.0

**Date :** 7 juillet 2026
**Objectif :** infrastructure open source complète + release (audit §6, §7 Phase 3)

## Réalisé

| Livrable | Contenu |
|---|---|
| `LICENSE` | MIT, copyright 2024-2026 Mandem (extrait du choix déjà présent dans le README — le fichier canonique manquait, bloquant juridique pour l'adoption). |
| `README.md` réécrit | Par patchs chirurgicaux préservant le contenu de l'auteur (théorie, 25 références, structure). Ajouts/mises à jour : badges (CI, licence, tests, black), section **Résultats clés** (tableau nb08 + 3 enseignements + vérification du biais 2dσ²), **Défi ML reformulé honnêtement** (leçon du leakage assumée publiquement), arbre d'architecture à jour, **section Applications aérospatial/aéronautique détaillée** (6 domaines : horloges GNSS, navigation inertielle quantique, QKD satellite, calcul quantique d'optimisation, QA de simulations, magnétométrie), correction physique entropie de von Neumann → entropie de Shannon de la distribution de Born, installation avec vérification pytest/CI, roadmap avec statuts réels, footer v0.3.0 juillet 2026. |
| `CONTRIBUTING.md` | Règles du projet (anti-leakage non négociable, frontière unique, logging, paths), style, workflow Git, PR. |
| Templates GitHub | Issues (bug, question scientifique), pull request avec checklist d'impact scientifique. |
| `ROADMAP.md` | Jalons 1-3 clos, jalon 4 détaillé (bruit corrélé, dérives de calibration, multiclasse — les régimes où le ML gagne sa place), jalon 5, table de dette technique avec efforts. |
| `CHANGELOG.md` | v0.1.0 → v0.3.0, format Keep a Changelog. |
| **Tag `v0.3.0`** | Release annotée : « le jalon de l'honnêteté scientifique ». |

## Décisions importantes

1. Licence MIT au nom de « Mandem » : respect du choix déjà exprimé dans le README de l'auteur (pas d'invention d'identité).
2. README **patché, pas remplacé** : les révisions GitHub de l'auteur (dont 2 commits du jour, rapatriés par fast-forward avant travaux) font foi ; seules les sections factuellement périmées ont été réécrites.
3. Le leakage du notebook 07 est raconté **publiquement dans le README** : c'est un argument pédagogique, pas une honte à cacher.
4. Correction scientifique visible : « entropie de von Neumann » était impropre (S(ρ)=0 pour tout état pur) — renommée entropie de Shannon de la distribution de Born, cohérente avec `entropy_shannon` dans le code.

## Reste à faire (post-session)

- L'utilisateur pousse : `git push origin main --tags`
- Créer la release GitHub depuis le tag v0.3.0 (interface web → Releases → Draft from tag, coller la section 0.3.0 du CHANGELOG)
- Vérifier le premier run CI au vert
- Demande utilisateur en attente de précision : « une refonte du code pour... » (message tronqué)

## Progression

Avancement global : ~78 % → **~88 %**. Le reste : jalon 4 (science avancée) et jalon 5 (API + interface pédagogique).
