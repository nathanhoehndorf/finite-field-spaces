import numpy as np
from typing import List

def compute_hamming_weight(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the Hamming weight for an array of vectors.
    Input shape: (M, n) -> Output shape (M,)
    """
    return np.count_nonzero(vectors, axis=1)

def generate_hamming_ball(universe: np.ndarray, center: np.ndarray, radius: int, linear_transform: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Generates a Hamming ball centered at `center` with a given `radius`
    under the basis defined by `linear_transform` over F_p^n
    
    B_{L}(v, r)= { x in F_p^n : weight( L(x-v) mod p ) <= r }
    """
    
    shifted = (universe - center) % p
    transformed = (shifted @ linear_transform.T) % p
    
    weights = compute_hamming_weight(transformed)
    mask = weights <= radius
    
    return universe[mask]

