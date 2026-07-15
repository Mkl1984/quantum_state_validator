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
- [ ] `qsv/data_generation.py` (docstrings/commentaires FR)
- [ ] `qsv/preprocessing.py`, `qsv/paths.py`, `tests/` (docstrings FR)
- [ ] `README.md` (FR → EN intégral)
- [ ] Notebooks 01-06 : alléger le markdown (convention « notebooks de code
  sobres ») + traduire les commentaires + ré-exécuter
- [ ] Notebooks 07-10 : markdown FR → EN + ré-exécution (figures avec
  labels FR à refaire)
- [ ] CONTRIBUTING.md, data/README.md, ROADMAP.md, CHANGELOG.md
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
