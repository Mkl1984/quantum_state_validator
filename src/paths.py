"""
Module: paths.py
Objectif: Chemins ancrés sur la racine du projet, indépendants du cwd.

Pourquoi ce module existe (audit A1)
------------------------------------
Les chemins relatifs comme ``"data/processed"`` dépendent du répertoire
courant au moment de l'exécution. Exécutés depuis ``notebooks/``, ils ont
créé un duplicata du dataset (1,8 Mo) et un dossier ``notebooks/models/``
parasites. Ce module ancre tous les chemins sur la racine du dépôt,
déterminée à partir de l'emplacement de ce fichier — invariant, lui,
quel que soit le cwd.

Usage
-----
>>> from src.paths import DATA_PROCESSED, MODELS_DIR
>>> df = pd.read_csv(DATA_PROCESSED / "quantum_states_10000.csv")
"""

from pathlib import Path

#: Racine du dépôt : ce fichier vit dans <racine>/src/, donc parents[1].
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

#: Dataset principal (10 000 états, dim=4, équilibré 50/50).
MAIN_DATASET: Path = DATA_PROCESSED / "quantum_states_10000.csv"
