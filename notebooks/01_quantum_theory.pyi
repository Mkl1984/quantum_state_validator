"""
Quantum Theory Module - Type Stubs
===================================

Ce fichier .pyi contient les signatures de types pour toutes les fonctions
du module de théorie quantique (01_quantum_theory.ipynb).

Objectif : Fournir une validation de types stricte et une documentation
           complète pour l'IDE et les type checkers (mypy, pyright).

Auteur  : MklZenin
Date    : 2024-11-12
Version : 1.0
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

# ============================================================================
# TYPE ALIASES - Définitions pour améliorer la lisibilité
# ============================================================================

# Type pour un état quantique (vecteur complexe)
QuantumState = NDArray[np.complex128]

# Type pour les probabilités (vecteur de réels positifs)
Probabilities = NDArray[np.float64]

# Type de retour pour la validation de normalisation
NormalizationResult = Tuple[bool, float]

# ============================================================================
# FONCTION 1 : Vérification de la normalisation d'un état quantique
# ============================================================================

def is_normalized(state: QuantumState, tolerance: float = 1e-6) -> NormalizationResult:
    """
    Vérifie si un état quantique est normalisé selon le critère ||ψ||² = 1.

    Cette fonction implémente la condition de normalisation fondamentale
    en mécanique quantique : la somme des probabilités de tous les états
    de base doit être égale à 1 (avec une tolérance numérique).

    Fondements théoriques
    ---------------------
    En mécanique quantique, un état pur |ψ⟩ doit satisfaire :
        ⟨ψ|ψ⟩ = Σᵢ |cᵢ|² = 1

    où cᵢ sont les amplitudes de probabilité et |cᵢ|² les probabilités
    de mesurer l'état dans la base |i⟩.

    Théorèmes invoqués
    ------------------
    - **Born Rule (Règle de Born)** : |⟨x|ψ⟩|² donne la densité de
      probabilité de trouver le système dans l'état |x⟩.
    - **Norme L² (Norme de Hilbert)** : Pour un espace de Hilbert,
      ||ψ||² = ⟨ψ|ψ⟩ doit converger.

    Paramètres
    ----------
    state : QuantumState (NDArray[np.complex128])
        Vecteur d'amplitudes complexes représentant l'état quantique.

        Shape : (n,) où n est la dimension de l'espace de Hilbert.

        Exemple : np.array([1/√2, 1/√2]) pour un état superposé équiprobable
                  np.array([0.6+0.3j, 0.5-0.4j]) pour un état complexe

    tolerance : float, optional (default=1e-6)
        Tolérance numérique pour la comparaison ||ψ||² ≈ 1.

        - Valeur par défaut : 10⁻⁶ (un millionième)
        - Justification : Prend en compte les erreurs d'arrondi en virgule
          flottante (floating-point precision)
        - Plus la tolérance est petite, plus le critère est strict

    Retourne
    --------
    is_valid : bool
        True si l'état est normalisé (||ψ||² ≈ 1 à tolerance près)
        False sinon

    norm_squared : float
        Valeur calculée de ||ψ||² = Σᵢ |cᵢ|²

        - Doit être proche de 1.0 pour un état valide
        - Si < 1 : "sous-normalisé" (probabilités insuffisantes)
        - Si > 1 : "sur-normalisé" (probabilités excédentaires)

    Notes mathématiques
    -------------------
    Le calcul effectué est :
        ||ψ||² = Σᵢ |cᵢ|² = Σᵢ (Re(cᵢ)² + Im(cᵢ)²)

    Pour un nombre complexe c = a + bi :
        |c|² = c* · c = (a - bi)(a + bi) = a² + b²

    où c* désigne le conjugué complexe de c.

    Complexité algorithmique
    ------------------------
    - Temps : O(n) où n = dimension de l'état
    - Espace : O(1) (calcul en place)

    Exemples
    --------
    >>> import numpy as np
    >>>
    >>> # État normalisé (superposition équiprobable)
    >>> state_valid = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    >>> is_normalized(state_valid)
    (True, 1.0)
    >>>
    >>> # État non normalisé
    >>> state_invalid = np.array([0.5, 0.5])
    >>> is_normalized(state_invalid)
    (False, 0.5)
    >>>
    >>> # État complexe normalisé
    >>> state_complex = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
    >>> is_normalized(state_complex)
    (True, 1.0)

    Voir aussi
    ----------
    normalize_state : Fonction pour normaliser un état invalide

    Références
    ----------
    - Born, M. (1926). "Zur Quantenmechanik der Stoßvorgänge"
    - Von Neumann, J. (1932). "Mathematical Foundations of Quantum Mechanics"
    """
    ...

# ============================================================================
# FONCTION 2 : Normalisation d'un état quantique
# ============================================================================

def normalize_state(state: QuantumState) -> QuantumState:
    """
    Normalise un état quantique pour satisfaire la condition ||ψ||² = 1.

    Cette fonction prend un état quantique arbitraire et le transforme
    en un état normalisé en divisant par sa norme. Les proportions relatives
    entre les amplitudes sont préservées.

    Fondements théoriques
    ---------------------
    Pour tout vecteur |ψ⟩ non nul, on peut construire l'état normalisé :
        |ψ_norm⟩ = |ψ⟩ / ||ψ||

    où ||ψ|| = √(⟨ψ|ψ⟩) = √(Σᵢ |cᵢ|²)

    Cette opération préserve la direction du vecteur dans l'espace de Hilbert
    tout en garantissant que sa norme soit exactement 1.

    Propriétés mathématiques
    ------------------------
    1. **Idempotence** : normalize(normalize(ψ)) = normalize(ψ)
    2. **Préservation des proportions** :
       |cᵢ|²/|cⱼ|² est identique avant et après normalisation
    3. **Linéarité scalaire** :
       normalize(λψ) = normalize(ψ) pour tout λ ≠ 0

    Théorème invoqué
    ----------------
    **Théorème de Gram-Schmidt (cas particulier)** : Tout vecteur non nul
    d'un espace préhilbertien peut être normalisé par division par sa norme.

    Paramètres
    ----------
    state : QuantumState (NDArray[np.complex128])
        État quantique à normaliser (peut être non normalisé).

        Shape : (n,) où n est la dimension de l'espace de Hilbert.

        IMPORTANT : Le vecteur ne doit pas être nul (||ψ|| ≠ 0)

    Retourne
    --------
    normalized_state : QuantumState (NDArray[np.complex128])
        État normalisé tel que ||ψ_norm||² = 1

        Shape : Identique à l'entrée (n,)

        Si l'état d'entrée était nul (||ψ|| = 0) :
        - Un warning est affiché
        - L'état inchangé est retourné (vecteur nul)

    Notes d'implémentation
    ----------------------
    Le calcul suit ces étapes :

    1. Calcul de la norme : ||ψ|| = √(Σᵢ |cᵢ|²)
    2. Vérification : ||ψ|| ≠ 0 ?
    3. Division : |ψ_norm⟩ = |ψ⟩ / ||ψ||

    La fonction np.sqrt() est utilisée car nous avons besoin de ||ψ||
    et non ||ψ||² (contrairement à is_normalized).

    Cas limites
    -----------
    - **État nul** : Si ||ψ|| = 0, retourne l'état inchangé avec warning
      (un état nul n'a pas de sens physique en MQ)
    - **Très petite norme** : Si ||ψ|| ≈ 0, risque d'instabilité numérique
      (division par un nombre proche de zéro)

    Complexité algorithmique
    ------------------------
    - Temps : O(n) pour le calcul de norme + O(n) pour la division = O(n)
    - Espace : O(n) pour le nouveau vecteur

    Exemples
    --------
    >>> import numpy as np
    >>>
    >>> # Normalisation d'un état simple
    >>> state = np.array([0.5, 0.5])
    >>> state_norm = normalize_state(state)
    >>> print(state_norm)
    [0.70710678 0.70710678]  # = [1/√2, 1/√2]
    >>>
    >>> # Vérification
    >>> is_normalized(state_norm)
    (True, 1.0)
    >>>
    >>> # État complexe
    >>> state_complex = np.array([0.6+0.3j, 0.5-0.4j])
    >>> state_complex_norm = normalize_state(state_complex)
    >>> print(f"Norme avant : {np.sqrt(np.sum(np.abs(state_complex)**2))}")
    Norme avant : 0.9273618495495704
    >>> print(f"Norme après : {np.sqrt(np.sum(np.abs(state_complex_norm)**2))}")
    Norme après : 1.0

    Attention
    ---------
    ⚠️  Cette fonction ne vérifie PAS que l'état normalisé est physiquement
        valide (ex: pas de contraintes sur la cohérence de phase).
        Elle garantit seulement ||ψ||² = 1.

    Voir aussi
    ----------
    is_normalized : Vérifier si un état est déjà normalisé

    Références
    ----------
    - Dirac, P.A.M. (1930). "The Principles of Quantum Mechanics"
    - Sakurai, J.J. (2017). "Modern Quantum Mechanics"
    """
    ...

# ============================================================================
# FONCTION 3 : Visualisation des probabilités quantiques
# ============================================================================

def visualize_probabilities(
    state: QuantumState, title: str = "Probabilités de mesure"
) -> None:
    """
    Visualise les probabilités |cᵢ|² d'un état quantique sous forme de
    diagramme en barres avec annotations détaillées.

    Cette fonction crée un graphique matplotlib montrant la distribution
    des probabilités de mesure dans chaque état de base, avec indication
    visuelle de la normalisation et statistiques.

    Fondements théoriques
    ---------------------
    Pour un état |ψ⟩ = Σᵢ cᵢ|i⟩, la probabilité de mesurer l'état de base
    |i⟩ est donnée par la règle de Born :
        P(i) = |⟨i|ψ⟩|² = |cᵢ|²

    La visualisation permet de vérifier intuitivement que Σᵢ P(i) = 1.

    Éléments visuels
    ----------------
    Le graphique comprend :
    1. **Barres de probabilité** :
       - Bleues (steelblue) si l'état est normalisé
       - Rouges (crimson) si l'état n'est pas normalisé

    2. **Ligne de référence** :
       - Ligne horizontale verte pointillée à y = 1.0
       - Représente la somme idéale Σᵢ |cᵢ|² = 1

    3. **Annotations numériques** :
       - Valeur exacte de chaque |cᵢ|² au-dessus de la barre
       - Format : 4 décimales pour la précision

    4. **Titre informatif** :
       - Statut : ✅ NORMALISÉ ou ❌ NON NORMALISÉ
       - Valeur de la somme Σ|cᵢ|² avec 6 décimales

    Paramètres
    ----------
    state : QuantumState (NDArray[np.complex128])
        État quantique dont on veut visualiser les probabilités.

        Shape : (n,) où n = nombre d'états de base

        Le graphique montrera n barres correspondant aux n états de base.

    title : str, optional (default="Probabilités de mesure")
        Titre principal du graphique.

        - Le statut de normalisation sera automatiquement ajouté
        - Exemple de titre final :
          "Probabilités de mesure\n✅ NORMALISÉ (Σ|cᵢ|² = 1.000000)"

    Retourne
    --------
    None
        La fonction affiche le graphique via plt.show() mais ne retourne rien.

        Le graphique est créé avec :
        - Taille : 10×6 pouces (figsize=(10, 6))
        - Style : 'seaborn-v0_8-darkgrid'
        - Grille : Affichée sur l'axe Y avec alpha=0.3

    Configuration du graphique
    --------------------------
    Axes :
    - **Axe X** : Indices des états de base |0⟩, |1⟩, |2⟩, ...
    - **Axe Y** : Probabilités |cᵢ|² ∈ [0, 1]

    Légende :
    - Ligne verte : "Somme idéale = 1.0"

    Notes d'utilisation
    -------------------
    Cette fonction utilise :
    - matplotlib.pyplot pour la visualisation
    - is_normalized() en interne pour vérifier la normalisation
    - np.abs(state)**2 pour calculer les probabilités

    Le graphique est automatiquement optimisé avec plt.tight_layout()
    pour éviter les chevauchements de labels.

    Cas d'usage typiques
    --------------------
    1. **Vérification visuelle** : Après normalisation, vérifier que
       les probabilités sont correctement distribuées

    2. **Comparaison d'états** : Visualiser plusieurs états pour comprendre
       les différences de distribution de probabilité

    3. **Débogage** : Identifier rapidement les états mal normalisés par
       la couleur rouge des barres

    4. **Pédagogie** : Illustrer la règle de Born et la normalisation
       dans un contexte d'enseignement

    Exemples
    --------
    >>> import numpy as np
    >>>
    >>> # État équiprobable sur 2 dimensions
    >>> state_eq = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    >>> visualize_probabilities(state_eq, "État équiprobable |+⟩")
    # Affiche 2 barres bleues de hauteur 0.5 chacune
    >>>
    >>> # État non normalisé
    >>> state_inv = np.array([0.5, 0.5])
    >>> visualize_probabilities(state_inv, "État non normalisé")
    # Affiche 2 barres ROUGES de hauteur 0.25 chacune
    # Titre indiquera "❌ NON NORMALISÉ (Σ|cᵢ|² = 0.500000)"
    >>>
    >>> # État complexe sur 3 dimensions
    >>> state_3d = np.array([0.6, 0.65, 0.46])
    >>> visualize_probabilities(state_3d, "État 3D")
    # Affiche 3 barres avec probabilités 0.36, 0.4225, 0.2116

    Dépendances requises
    --------------------
    - numpy : Calculs numériques
    - matplotlib.pyplot : Visualisation
    - is_normalized : Fonction interne pour vérifier la normalisation

    Voir aussi
    ----------
    is_normalized : Vérifier la normalisation d'un état
    normalize_state : Normaliser un état avant visualisation

    Références
    ----------
    - Nielsen & Chuang (2010). "Quantum Computation and Quantum Information"
      (Section sur la mesure et la règle de Born)
    """
    ...

# ============================================================================
# INFORMATIONS ADDITIONNELLES
# ============================================================================

__all__ = [
    "is_normalized",
    "normalize_state",
    "visualize_probabilities",
    # Types exports
    "QuantumState",
    "Probabilities",
    "NormalizationResult",
]

# Version du module
__version__ = "1.0.0"

# Metadata
__author__ = "MklZenin"
__date__ = "2024-11-12"
__module__ = "quantum_theory"
