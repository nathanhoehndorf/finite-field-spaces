import numpy as np

def compute_sumset(set_elements: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Computes the unique elements of the sumset S + S over F_p
    Optimized via broadcasting
    """
    if len(set_elements) == 0:
        return np.empty((0, set_elements.shape[1]), dtype=np.int8)
    
    broadcasted_sum = (set_elements[:, np.newaxis, :] + set_elements[np.newaxis, :, :]) % p
    reshaped_sums = broadcasted_sum.reshape(-1, set_elements.shape[1])
    
    return np.unique(reshaped_sums, axis=0)

def find_maximum_subspace_dimension(sumset: np.ndarray, p: int = 2) -> int:
    """
    Finds the dimension of the largest linear subspace completely contained within S+S.
    Uses an iterative greedy check of linearly independent generators.
    """
    n = sumset.shape[1]
    sumset_set = {tuple(row) for row in sumset}
    
    # Origin must be in sumset for valid subspace
    if tuple(np.zeros(n, dtype=np.int8)) not in sumset_set:
        return -1
    
    independent_generators = []
    
    for candidate in sumset:
        if np.all(candidate == 0):
            continue
        tuple_cand = tuple(candidate)
        
        valid_generator = True
        current_span = [np.zeros(n, dtype=np.int8)]
        
        for gen in independent_generators:
            current_span += [(gen * k + s) % p for k in range(1,p) for s in current_span]
            
        for vec in current_span:
            for k in range(1,p):
                test_vec = (vec + k * candidate) % p
                if tuple(test_vec) not in sumset_set:
                    valid_generator = False
                    break
            if not valid_generator:
                break
            
        if valid_generator:
            # Verify linear independence
            test_matrix = np.array(independent_generators + [candidate])
            if np.linalg.matrix_rank(test_matrix.astype(float)) == len(independent_generators) + 1:
                independent_generators.append(candidate)
                
    return len(independent_generators)

