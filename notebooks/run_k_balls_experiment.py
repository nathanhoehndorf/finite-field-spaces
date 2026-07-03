import time
import json
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core import generate_space, generate_random_basis
from src.geometries import generate_hamming_ball
from src.covers import generate_covering, complement
from src.operators import compute_sumset, find_maximum_subspace_dimension

np.random.seed(0)

def run_experiments(ns=(5,6), Ks=(4,5,6,7,8,9), trials=10000, radius=1, p=2, show_progress=True):
    results = {}
    zero_hole_records = []
    for n in ns:
        universe = generate_space(n, p)
        results[n] = {}
        for K in Ks:
            if show_progress:
                print(f'Running n={n}, K={K}, trials={trials}')
            res_list = []
            start = time.time()
            for t in range(trials):
                bases = [generate_random_basis(n, p) for _ in range(K)]
                centers = [universe[np.random.choice(len(universe))] for _ in range(K)]
                try:
                    covered = generate_covering(centers, radius, bases=bases, p=p, universe=universe)
                except Exception:
                    # fallback to previous behaviour if generation fails for a trial
                    continue

                S = complement(universe, covered)
                S_size = len(S)
                if S_size == 0:
                    max_dim = None
                    zero_hole_records.append({'n': n, 'K': K, 'trial': t, 'centers': [list(map(int,c)) for c in centers]})
                else:
                    S_plus_s = compute_sumset(S, p)
                    max_dim = find_maximum_subspace_dimension(S_plus_s, p)
                res_list.append({'S_size': S_size, 'max_subspace_dim': max_dim})
                if show_progress and (t+1) % 200 == 0:
                    elapsed = time.time() - start
                    print(f'  trial {t+1}/{trials} elapsed={elapsed:.1f}s')
            results[n][K] = res_list
    return results, zero_hole_records

if __name__ == '__main__':
    out_dir = Path(__file__).parent
    exp_results, zero_holes = run_experiments(ns=(16,), Ks=(3,), trials=10000, radius=2, p=2, show_progress=True)
    summary = {}
    for n, Ks_res in exp_results.items():
        summary[n] = {}
        for K, records in Ks_res.items():
            S_sizes = [r['S_size'] for r in records]
            dims = [r['max_subspace_dim'] for r in records if r['max_subspace_dim'] is not None]
            summary[n][K] = {
                'trials_run': len(S_sizes),
                'S_size_mean': float(np.mean(S_sizes)) if len(S_sizes)>0 else None,
                'S_size_median': float(np.median(S_sizes)) if len(S_sizes)>0 else None,
                'S_size_zero_count': int(sum(1 for s in S_sizes if s==0)),
                    'max_subspace_dim_mean': float(np.mean(dims)) if len(dims)>0 else None,
                    'max_subspace_dim_median': float(np.median(dims)) if len(dims)>0 else None,
                    'max_subspace_dim_min': int(min(dims)) if len(dims)>0 else None,
            }
    out_path = out_dir / 'k_balls_experiment_results.npz'
    np.savez_compressed(out_path, exp_results=exp_results, zero_holes=zero_holes, summary=summary)
    print(f'Results saved to {out_path}')
