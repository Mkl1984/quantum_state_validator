# Releasing quantum-state-validator to PyPI

Manual steps (require the maintainer's PyPI account; never automated with
stored credentials in this repository).

## Preconditions
1. CI green on main; `pytest tests/ -q` locally green.
2. Version bumped in `src/qsv/__init__.py` (single source; pyproject reads it).
3. CHANGELOG section cut for the version; tag `vX.Y.Z` pushed.

## Build and verify (already scripted below, run from the repo root)
```bash
python -m pip install --upgrade build twine
python -m build                 # dist/*.whl + dist/*.tar.gz
python -m twine check dist/*    # both must PASS
```

## Publish
```bash
python -m twine upload -r testpypi dist/*   # rehearsal on test.pypi.org
pip install -i https://test.pypi.org/simple/ quantum-state-validator  # smoke test
python -m twine upload dist/*               # the real thing
```

## After publishing
- Create the GitHub release from the tag (paste the CHANGELOG section).
- Verify `pip install quantum-state-validator` then
  `python -c "import qsv; print(qsv.__version__)"` and `qsv --version`.

Status 2026-07-16: artifacts for 0.5.0 built and twine-checked (PASSED).
Publication deliberately waits for the v0.5.0 tag, itself gated on the
web-app acceptance report (reports/EMERGENT_ACCEPTANCE.md).
