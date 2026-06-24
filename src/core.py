import numpy as np
import itertools

def generate_space(n: int, p: int = 2) -> np.ndarray:
    """
    Generates all p^n vectors in the vector space F_p^n.
    Returns a 2D numpy array of shape (p^n, n).
    """
    elements = list(itertools.product(range(p), repeat=n))
    return np.array(elements, dtype=np.int8)

def is_invertible(matrix: np.ndarray, p: int=2) -> bool:
    """Checks if an n x n matrix is invertible over F_p."""
    det = int(round(np.linalg.det(matrix)))
    return np.gcd(det, p) == 1

def generate_random_basis(n: int, p: int = 2) -> np.ndarray:
    """Generates a random invertible n x n matrix over F_p to represent a basis change."""
    while True:
        matrix = np.random.randint(0, p, size=(n,n)).astype(np.int8)
        if is_invertible(matrix, p):
            return matrix