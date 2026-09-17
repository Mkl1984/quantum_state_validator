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

### ✅ Jalon 4 — Là où le ML gagne sa place (v0.4.0) — CLOS

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

**4c (fait, notebook 11)** :
- ✅ Préparation d'états cibles *connus* (`src/preparation.py`, en anglais) —
  la référence directionnelle brise la limite d'isotropie du nb 09 :
  `rotated` diagnostiqué à 100 % (vs 59 % de rappel sans référence).
  Chaque statistique seule plafonne à 0.667 (aveugle à une classe) ; la
  paire (norme, fidélité) atteint 0.92 avec un arbre de profondeur 3 et
  bat le RF sur amplitudes brutes — 4e leçon « représentation > algorithme ».

**4d (fait, notebook 12)** :
- ✅ Dimensions variables : pipeline d-agnostique vérifié (d ∈ {2,4,8,16},
  tests paramétrés). Résultat contre-intuitif documenté : la validation
  devient PLUS facile quand d croît (concentration de la mesure), avec
  le caveat honnête sur la population d'invalides du générateur.
- ✅ Abaque N ↔ taux d'erreur : FPR 19 % à N=25 → 1.1 % à N=1600 →
  plancher à N≈6400 (marge 0.05, d=4). Énoncé de dimensionnement :
  FPR ≤ 1 % exige N ≈ 2000 ; marge et budget s'échangent comme stabilité
  et temps d'intégration dans un budget d'horloge GNSS.

**Le jalon 4 est clos (v0.4.0).**

## Phase finale — état d'avancement

**Fait** :
- ✅ Notebook 13 `13_project_report.ipynb` (EN, exécuté) : rapport de projet
  complet + carnet de bord des erreurs à la première personne (section 7),
  construit exclusivement sur reports/ et l'historique Git.
- ✅ Traduction EN : `qsv/features.py` (module central), en plus des modules
  nés en anglais (validators, api, preparation).

**Checklist de traduction restante** (ordre de priorité) :
- [x] `qsv/data_generation.py` — traduit, **dataset régénéré au hash MD5
  identique** (preuve que la logique est intacte)
- [x] `qsv/preprocessing.py`, `qsv/paths.py`, `tests/` — traduits, 57 tests
  collectés et verts, zéro caractère accentué restant dans le code
- [x] `README.md` — traduit intégralement (580 lignes, structure et 25 références préservées)
- [x] Notebooks 01-06 : reconstruits sobres et en anglais (versions FR dans
  l'historique Git), ré-exécutés — le 04 reproduit toujours le CSV octet
  pour octet
- [x] Notebooks 07-10 : avertissement du 07 traduit (sorties d'époque
  préservées comme archive) ; 08-10 retraduits et ré-exécutés — chiffres
  identiques (seeds déterministes)
- [x] CONTRIBUTING.md, data/README.md — traduits et mis à jour (la note
  périmée « norm_squared utile pour le ML » devient l'avertissement leakage)
- Note : ROADMAP.md et CHANGELOG.md restent en français — documents de
  pilotage interne, à traduire en dernier si souhaité.
- Note : les rapports de session dans reports/ antérieurs au 2026-07-08
  restent en français — ce sont des archives historiques datées.

## Conventions du projet (directives du 2026-07-07)

- **Langue : anglais partout** — code, commentaires, figures, documentation.
  Le nouveau code est en anglais ; la traduction de l'existant (docstrings
  et notebooks en français) fait partie de la phase finale de mise en forme.
- **Notebooks de code : sobres** — implémentation et commentaires courts ;
  la documentation détaillée vit dans le notebook final de projet.
- **Phase finale (après les jalons)** : notebook de documentation complet
  (présentation, théorie, choix méthodologiques, architecture, résultats,
  conclusion) incluant un retour d'expérience à la première personne sur
  les erreurs réellement rencontrées (matière première : reports/ et
  l'historique Git — rien d'inventé) ; traduction intégrale en anglais ;
  pas d'emoji ; style naturel sans sur-ingénierie.

### 🔄 Jalon 5 — Production et interface (v1.0.0) — EN COURS

**P0 distribution (fait, 16 juil. 2026)** : ✅ `qsv.adapters` (Qiskit/
PennyLane/Cirq par duck-typing, zéro dépendance framework), ✅ CLI
`qsv validate` (npy/csv, codes de sortie type pytest — le garde-fou CI),
✅ artefacts PyPI 0.5.0 construits et vérifiés (twine PASSED) — publication
en attente du tag v0.5.0 (voir docs/RELEASING.md). ✅ `qsv.density`
(hermiticité/trace/positivité, mode bruité calibré empiriquement — le fait
tomographique des petites valeurs propres négatives est géré, notebook 14).
✅ `qsv.tomography` (17 sept. 2026) — modèle de comptage exact (Poisson) qui
remplace la simplification gaussienne, dernière limite scientifique déclarée
du projet. **Le P0 code est clos.**

Ce que l'expérience a montré (notebook 15, chiffres avant interprétation) :
- l'estimateur de comptage est **non biaisé** : sur 200 000 répétitions
  (d ∈ {2,4,8}, N ∈ {100,1000}), le plus grand biais mesuré vaut 4,5e-4, soit
  environ deux erreurs de Monte-Carlo (résolution 2,2e-4), et les douze mesures
  se dispersent autour de zéro sans signe systématique ; la dispersion suit
  sqrt(||psi||²/N) à 0,3 % près ;
- le terme 2dσ² de `validators.py` est un **artefact du modèle gaussien**, pas
  un fait de comptage : le biais gaussien mesuré suit 2dσ² sur tout (d, N)
  testé (+0.08018 contre 0.08000 à d=16, N=100) tandis que celui du comptage
  ne dépasse jamais 2,2e-4 — sa propre résolution de Monte-Carlo — et surtout
  **ne croît pas avec d** : c'est l'observation discriminante, le biais gaussien
  est proportionnel à d, celui du comptage à rien. La correction reste juste
  *dans son modèle* — elle est conservée dans `validators.py` pour cette
  raison ;
- **les deux modèles s'accordent sur la dispersion** (sd 0.0498 contre 0.0500
  à d=4, N=400) : le choix σ = 1/(2√N) du notebook 08 était bon pour la
  quantité qui pilote réellement la décision ;
- **l'abaque du notebook 12 survit** : l'écart de FPR ne dépasse jamais
  0,6 point et celui de FNR 1,7 point, les deux maximaux au plus petit budget
  où tous les taux sont de toute façon dominés par le bruit ; aucun biais
  systématique dans une direction. Les
  jalons 1 à 4 tiennent tels que publiés. Quatrième résultat « négatif »
  conservé plutôt qu'enterré.
- **non-identifiabilité prouvée** : conditionnées au total, les fréquences
  k/Σk coïncident à 3 décimales pour ||psi||² ∈ {0.85, 1.00, 1.20} (la 4e
  décimale bouge encore d'une unité : bruit de comptage résiduel à N = 400 000,
  pas dépendance résiduelle à la norme) alors que les totaux diffèrent de 41 %. Toute l'information de norme vit dans le
  comptage total — une vérification de norme exige une **exposition calibrée**.
  C'est l'invariance d'échelle du notebook 07 vue depuis l'autre bout.

Conséquence documentaire : la section « hypothèses et limites » du notebook 13
ne liste plus le bruit gaussien comme limite mais comme approximation mesurée ;
la limite ouverte principale devient la **population d'invalides synthétique**
(états adverses collés à la marge).

**5a (fait)** : ✅ API FastAPI (`src/api.py`, EN) — décision d'architecture
issue du jalon 4 : l'API sert les validateurs GAGNANTS (test à seuil corrigé
du biais, paire norme/fidélité), pas de modèle ML entraîné. Endpoints
`/validate` (modes exact/bruité, avertissement de budget insuffisant fondé
sur l'abaque du nb 12) et `/preparation-qa` (moniteur deux canaux du nb 11),
réponses avec explications pédagogiques, docs OpenAPI générées. 7 tests.

**5b (fait)** : ✅ Double mode d'utilisation — paquet installable
`quantum-state-validator` (pyproject, src-layout, `pip install -e .`) :
la logique de décision vit dans `qsv/validators.py` (pur, sans HTTP),
importable dans tout projet (`from qsv import validate_state,
preparation_qa`) ; l'API devient une enveloppe mince du même code.
Les deux modes sont combinables par construction. Notebooks rebranchés
sur le paquet et ré-exécutés (le 04 reproduit toujours le CSV octet
pour octet).

**5c (reste)** :
- Interface pédagogique interactive (parcours guidés, quiz adaptatifs,
  validation étape par étape — voir la vision du projet)
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
