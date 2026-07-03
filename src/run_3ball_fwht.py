import time
import numpy as np
import itertools
import argparse
from multiprocessing import Pool, cpu_count

from ffspaces.core import generate_random_basis
from ffspaces.fwht_operators import compute_sumset_fwht, vectors_to_ints, ints_to_vectors
from ffspaces.operators import find_maximum_subspace_dimension
from ffspaces.covers import generate_covering, complement

def generate_standard_ball(n: int, r: int) -> np.ndarray:
    """
    Generates all vectors in F_2^n with Hamming weight <= r
    """
    vectors = []
    for weight in range(r+1):
        for combo in itertools.combinations(range(n), weight):
            vec = np.zeros(n, dtype=np.int8)
            vec[list(combo)] = 1
            vectors.append(vec)
    return np.array(vectors, dtype=np.int8)
    
def invert_matrix_f2(M: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a matrix over F_2 using Gaussian elimination.
    """
    n = M.shape[0]
    A = np.hstack([M, np.eye(n, dtype=np.int8)]).astype(np.int8) % 2
    for i in range(n):
        if A[i, i] == 0:
            for j in range(i+1, n):
                if A[j,i] == 1:
                    A[[i,j]] = A[[j,i]]
                    break
        for j in range(n):
            if i != j and A[j,i] == 1:
                A[j] ^= A[i]
    return A[:, n:] % 2

def worker_trial(args):
    """
    Single trial execution for multiprocessing
    args: (n, r, seed, exhaustive)
    """
    if len(args) == 3:
        n, r, seed = args
        exhaustive = False
    else:
        n, r, seed, exhaustive = args
    np.random.seed(seed)

    universe_size = 1 << n

    W_r = generate_standard_ball(n, r)
    L1 = generate_random_basis(n, 2)
    L2 = generate_random_basis(n, 2)
    L1_inv_T = invert_matrix_f2(L1).T
    L2_inv_T = invert_matrix_f2(L2).T

    c1 = np.random.randint(0, 2, n, dtype=np.int8)
    c2 = np.random.randint(0, 2, n, dtype=np.int8)
    c3 = np.random.randint(0, 2, n, dtype=np.int8)

    # Use consolidated covering generator (fast binary path)
    centers = [c1, c2, c3]
    bases = [None, L1_inv_T, L2_inv_T]
    covered = generate_covering(centers, r, bases=bases, p=2, universe=None)
    covered_ints = vectors_to_ints(covered)

    S_size = universe_size - len(covered_ints)

    max_dim = None
    if S_size > 0:
        all_ints = np.arange(universe_size, dtype=np.int32)
        mask = np.ones(universe_size, dtype=bool)
        mask[covered_ints] = False
        S_ints = all_ints[mask]

        # We're cheating a little bit here
        S_vectors = ints_to_vectors(S_ints, n)

        S_plus_S_vectors = compute_sumset_fwht(S_vectors)

        if exhaustive:
            # use exact search for max subspace dimension
            max_dim = find_maximum_subspace_dimension(S_plus_S_vectors, p=2, exhaustive=True)
        else:
            if len(S_plus_S_vectors) == universe_size:
                max_dim = n
            else:
                max_dim = int(np.log2(len(S_plus_S_vectors)))

    return {'S_size': S_size, 'max_subspace_dim': max_dim}

def run_multithreaded_sweep(n=16, r_vals=[4,5,6], trials_per_r=100, exhaustive=False):
    print(f"Starting 3-Ball Independent Basis Sweep for n={n}")
    print(f"Universe Size; {1 << n}")

    results = {}
    num_cores = cpu_count()

    with Pool(processes=num_cores) as pool:
        for r in r_vals:
            print(f"\n--- Running r={r} (Codimension bound tests) ---")
            start = time.time()

            seeds = np.random.randint(0, 1000000, trials_per_r)
            tasks = [(n, r, seed, exhaustive) for seed in seeds]

            trial_results = pool.map(worker_trial, tasks)

            S_sizes = [res['S_size'] for res in trial_results]
            dims = [res['max_subspace_dim'] for res in trial_results if res['max_subspace_dim'] is not None]

            perfect_coverings = sum(1 for s in S_sizes if s == 0)
            full_space_sumsets = sum(1 for d in dims if d == n)

            print(f"Trials: {trials_per_r} | Time: {time.time() - start:.2f}s")
            print(f"Perfect Coverings (S is empty): {perfect_coverings}")
            if len(dims) > 0:
                print(f"Trials where S+S = F_2^n: {full_space_sumsets} / {len(dims)}")
                print(f"Minimum dimension of S+S observed: {min(dims)}")

            results[r] = trial_results
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3-ball FWHT sweep')
    parser.add_argument('--n', type=int, default=20, help='dimension n')
    parser.add_argument('--r', type=int, nargs='+', default=[4,5,6], help='r values to sweep')
    parser.add_argument('--trials', type=int, default=100, help='trials per r')
    parser.add_argument('--exhaustive', action='store_true', help='Use exhaustive search for max subspace dimension')

    args = parser.parse_args()

    run_multithreaded_sweep(n=args.n, r_vals=args.r, trials_per_r=args.trials, exhaustive=args.exhaustive)