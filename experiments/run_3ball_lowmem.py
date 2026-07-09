"""
run_3ball_lowmem.py
-------------------
Test whether, after removing 3 Hamming balls from F_2^n, the remainder set S
satisfies S+S = F_2^n.  Designed for n > 20 (up to 25), using low-memory
integer-encoded operations that never materialise the full vector array.

Memory footprint per trial (p=2):
  n=21 ~32 MB  |  n=22 ~64 MB  |  n=23 ~128 MB
  n=24 ~256 MB |  n=25 ~512 MB

Usage examples:
    python experiments/run_3ball_lowmem.py --n 22 --r 5 --trials 50
    python experiments/run_3ball_lowmem.py --n_min 21 --n_max 23 --r 4 5 6 --trials 20
    python experiments/run_3ball_lowmem.py --n 25 --r 6 --trials 10 --workers 1
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from ffspaces.core import generate_random_basis
from ffspaces.fwht_operators import vectors_to_ints
from ffspaces.lowmem import (
    complement_ints_lowmem,
    compute_sumset_lowmem_from_ints,
    generate_covering_ints_lowmem,
)


# ---------------------------------------------------------------------------
# F_2 matrix inverse (needed to get L^{-T} for basis transforms)
# ---------------------------------------------------------------------------

def _invert_f2(M: np.ndarray) -> np.ndarray:
    """Invert an n×n matrix over F_2 via Gauss-Jordan elimination."""
    n = M.shape[0]
    A = np.hstack([M, np.eye(n, dtype=np.int8)]).astype(np.int8) % 2
    for i in range(n):
        if A[i, i] == 0:
            for j in range(i + 1, n):
                if A[j, i] == 1:
                    A[[i, j]] = A[[j, i]]
                    break
        for j in range(n):
            if i != j and A[j, i] == 1:
                A[j] ^= A[i]
    return A[:, n:] % 2


# ---------------------------------------------------------------------------
# Single-trial worker (suitable for multiprocessing.Pool.map)
# ---------------------------------------------------------------------------

def worker_trial(args: tuple) -> dict:
    """Run one random trial and return a result dict.

    args: (n, r, seed)

    The 3 balls are:
      Ball 1 – standard basis, random centre c1
      Ball 2 – random invertible basis L1, random centre c2
      Ball 3 – random invertible basis L2, random centre c3

    Returns
    -------
    dict with keys:
      'n'           : dimension
      'r'           : radius
      'covered'     : number of elements in the union of the 3 balls
      'S_size'      : |F_2^n \ covered|
      'S_plus_S_full': True iff S+S == F_2^n  (or S is empty)
      'S_plus_S_size': |S+S|
    """
    n, r, seed = args
    rng = np.random.default_rng(seed)
    universe_size = 1 << n

    # Random centres
    c1 = rng.integers(0, 2, n, dtype=np.int8)
    c2 = rng.integers(0, 2, n, dtype=np.int8)
    c3 = rng.integers(0, 2, n, dtype=np.int8)

    # Random invertible bases; ball membership is measured via L^{-T}
    # (same convention as run_3ball_fwht.py)
    L1 = generate_random_basis(n, 2, rng=rng)
    L2 = generate_random_basis(n, 2, rng=rng)
    L1_inv_T = _invert_f2(L1).T
    L2_inv_T = _invert_f2(L2).T

    # Build covering as sorted integer indices (no vector arrays allocated)
    covered_ints = generate_covering_ints_lowmem(
        n=n,
        centers=[c1, c2, c3],
        radii=r,
        bases=[None, L1_inv_T, L2_inv_T],
        p=2,
    )
    n_covered = len(covered_ints)
    S_size = universe_size - n_covered

    if S_size == 0:
        # Balls perfectly tile F_2^n; S+S is vacuously empty
        return {
            'n': n, 'r': r,
            'covered': n_covered, 'S_size': 0,
            'S_plus_S_full': True, 'S_plus_S_size': 0,
        }

    # Complement: S = F_2^n \ covered  (uses a bool mask, 1 byte/element)
    S_ints = complement_ints_lowmem(n, covered_ints, p=2)
    del covered_ints  # release before FWHT allocation

    # S+S via in-place FWHT (peak ≈ 2 × float64 arrays of size 2^n)
    SS_ints = compute_sumset_lowmem_from_ints(S_ints, n, p=2)
    del S_ints

    SS_size = len(SS_ints)
    is_full = (SS_size == universe_size)

    return {
        'n': n, 'r': r,
        'covered': n_covered, 'S_size': S_size,
        'S_plus_S_full': is_full, 'S_plus_S_size': SS_size,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def _max_workers_for_n(n: int, requested: int | None) -> int:
    """Cap worker count to avoid OOM for large n.

    Approximate peak RAM per worker (float64 FWHT):
      n=21 ~32MB | n=22 ~64MB | n=23 ~128MB | n=24 ~256MB | n=25 ~512MB
    """
    caps = {21: cpu_count(), 22: cpu_count(), 23: 8, 24: 4, 25: 2}
    cap = caps.get(n, max(1, cpu_count() // 2))
    hw = min(cpu_count(), cap)
    if requested is not None:
        return max(1, min(requested, cap))
    return max(1, hw)


def run_sweep(
    n: int,
    r_vals: list[int],
    trials: int,
    seed: int,
    workers: int | None,
) -> dict:
    """Run a full (n, r) sweep, returning results keyed by r."""
    nw = _max_workers_for_n(n, workers)
    print(f"\n{'='*60}")
    print(f"n={n}  |  |F_2^n|={1<<n:,}  |  workers={nw}")
    print(f"{'='*60}")

    all_results: dict[int, list[dict]] = {}
    rng_top = np.random.default_rng(seed)

    for r in r_vals:
        seeds = rng_top.integers(0, 1 << 31, trials)
        tasks = [(n, r, int(s)) for s in seeds]

        t0 = time.perf_counter()
        if nw > 1:
            with Pool(processes=nw) as pool:
                results = pool.map(worker_trial, tasks)
        else:
            results = [worker_trial(t) for t in tasks]
        elapsed = time.perf_counter() - t0

        S_sizes = [res['S_size'] for res in results]
        SS_full = [res['S_plus_S_full'] for res in results]
        SS_sizes = [res['S_plus_S_size'] for res in results if res['S_size'] > 0]

        n_perfect = sum(1 for s in S_sizes if s == 0)
        n_ss_full = sum(1 for f in SS_full if f)

        print(f"\nr={r}  ({trials} trials, {elapsed:.1f}s)")
        print(f"  Perfect coverings (S=∅):            {n_perfect} / {trials}")
        if trials - n_perfect > 0:
            non_empty = trials - n_perfect
            print(f"  Trials with S non-empty:             {non_empty}")
            print(f"  S+S = F_2^n:                         {n_ss_full - n_perfect} / {non_empty}")
            print(f"  Min |S|:   {min(s for s in S_sizes if s > 0):,}")
            print(f"  Mean |S|:  {np.mean([s for s in S_sizes if s > 0]):,.0f}")
            if SS_sizes:
                print(f"  Min |S+S|: {min(SS_sizes):,}  (full = {1<<n:,})")
        all_results[r] = results

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3-ball low-memory sweep: test S+S=F_2^n after removing 3 Hamming balls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n", type=int, default=None,
                       help="Single dimension to test.")
    group.add_argument("--n_min", type=int, default=21,
                       help="Minimum n (inclusive) for range sweep.")
    parser.add_argument("--n_max", type=int, default=22,
                        help="Maximum n (inclusive) for range sweep (ignored when --n is set).")
    parser.add_argument("--r", type=int, nargs="+", default=[4, 5, 6],
                        help="Ball radius / radii to test.")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of random trials per (n, r) pair.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master RNG seed.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (auto-capped for large n).")

    args = parser.parse_args()

    if args.n is not None:
        n_vals = [args.n]
    else:
        n_vals = list(range(args.n_min, args.n_max + 1))

    for n in n_vals:
        if n < 1:
            print(f"Skipping n={n} (must be >= 1)")
            continue
        run_sweep(
            n=n,
            r_vals=sorted(args.r),
            trials=args.trials,
            seed=args.seed,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
