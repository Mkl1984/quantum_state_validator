# Contribuer à Quantum State Validator

Merci de votre intérêt ! Ce projet est un environnement d'apprentissage de la
mécanique quantique et du ML : la **rigueur scientifique et la pédagogie
priment sur tout le reste**. Une contribution qui rend le code plus rapide
mais moins compréhensible sera refusée ; une contribution qui documente une
erreur instructive sera célébrée (voir le notebook 07).

## Mise en place

```bash
git clone https://github.com/Mkl1984/quantum_state_validator.git
cd quantum_state_validator
python -m venv venv && source venv/bin/activate   # .\venv\Scripts\Activate.ps1 sous Windows
pip install -r requirements.txt
pytest tests/ -q          # 28 tests doivent passer avant de commencer
```

## Règles du projet

1. **Aucun target leakage.** Toute nouvelle feature d'entraînement doit être
   classée invariante ou sensible à l'échelle dans `src/features.py`, avec un
   test d'invariance si elle prétend être invariante. Le test
   `test_invariant_features_are_scale_invariant` est non négociable.
2. **La frontière de classe est unique** : |‖ψ‖² − 1| ≥ `norm_margin` pour tout
   état invalide (voir `data/README.md`). Ne la contournez jamais localement.
3. **Les bibliothèques ne parlent pas** : `logging`, jamais `print()` dans `src/`.
4. **Chemins via `src/paths.py`** — jamais de chemins relatifs au cwd.
5. **Chaque fonction publique** : docstring avec formules disséquées et leur
   lecture en français, hypothèses, et signification physique du résultat.
6. **Chaque changement de comportement** : un test qui le verrouille.

## Style et qualité

- Formatage : `black src/ tests/` (vérifié en CI, bloquant)
- Tests : `pytest tests/ -q` (bloquant en CI, Python 3.10 et 3.12)
- Notebooks : exécutés de bout en bout avant commit, outputs inclus

## Workflow Git

- Branches : `feat/<sujet>`, `fix/<sujet>`, `docs/<sujet>` depuis `main`
- Commits : [Conventional Commits](https://www.conventionalcommits.org/) en
  anglais — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`
- Un commit = un changement logique. Le message explique le *pourquoi*.
- Pull request vers `main` : la CI doit être verte ; décrivez l'impact
  scientifique éventuel (features, frontière de classe, modèle de bruit)

## Signaler un problème

Ouvrez une issue avec : version Python, OS, étapes de reproduction minimales,
comportement attendu vs observé. Pour une question scientifique (formulation,
modèle de bruit, features), citez la section du README ou du notebook concernée.

## Licence

En contribuant, vous acceptez que votre contribution soit publiée sous
licence MIT (voir LICENSE).
