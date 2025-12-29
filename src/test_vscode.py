import numpy as np


def test_function():
    """Test si VS Code détecte les erreurs et formate le code."""
    # Cette ligne est volontairement trop longue pour tester le formatage automatique avec Black
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    
    # Variable non utilisée (Pylint devrait le signaler)
    unused_variable = 42
    
    print("Hello from VS Code!")
    
    return x.mean()


if __name__ == "__main__":
    result = test_function()
    print(f"Moyenne: {result}")