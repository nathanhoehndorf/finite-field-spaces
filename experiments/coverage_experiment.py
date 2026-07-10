#!/usr/bin/env python3
"""
coverage_experiment.py
----------------------
Sweeps coverage density |S+S| / 2^n after removing 3 random Hamming balls
of radius r from F_2^n.

  Baseline  n = 10, 12, 14, 16, 18
  Big       n = 20, 22, 24, 26    ← low-memory FWHT path

Ball generation is fully vectorised (chunk-wise uint32 popcount / GF(2) parity),
avoiding slow Python-level iteration for large r.

Peak RAM per worker (FWHT float64 arrays):
  n ≤ 22  ~few MB  | n=24 ~256 MB | n=25 ~512 MB | n=26 ~1 GB

Outputs
-------
  • Per-trial stdout table:  n, r, |S|, |S+S|, |S+S|/2^n
  • Analysis:
      [1]  min(n) where |S| ≥ 2^(n/2)  but  |S+S| < 2^n
      [2]  d(mean_density)/dr at consecutive r steps
  • experiments/coverage_density_heatmap.png
  • experiments/coverage_experiment_results.npz

Usage
-----
    python experiments/coverage_experiment.py
    python experiments/coverage_experiment.py --baseline_trials 30 --big_trials 10
    python experiments/coverage_experiment.py --skip_big --baseline_trials 50
    python experiments/coverage_experiment.py --workers 2 --big_trials 3
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from math import comb
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')           # headless; change to 'TkAgg' for an interactive window
import matplotlib.pyplot as plt

# ── make ffspaces importable when run directly from the repo ────────────────
_SRC = Path(__file__).resolve().parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ffspaces.core import generate_random_basis
from ffspaces.lowmem import compute_sumset_lowmem_from_ints


# ══════════════════════════════════════════════════════════════════════════════
# Experiment configurations
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_CONFIG: List[Tuple[int, List[int]]] = [
    (10, [2, 3, 4]),
    (12, [2, 3, 4, 5]),
    (14, [3, 4, 5, 6]),
    (16, [3, 4, 5, 6, 7]),
    (18, [4, 5, 6, 7, 8]),
]

BIG_CONFIG: List[Tuple[int, List[int]]] = [
    (20, [3, 4, 5, 6, 7, 8, 9]),
    (22, [3, 4, 6, 8, 10]),
    (24, [3, 4, 6, 8, 10, 11]),
    (26, [3, 5, 7, 9, 11, 12]),
]


# ══════════════════════════════════════════════════════════════════════════════
# GF(2) matrix inverse (Gauss-Jordan)
# ══════════════════════════════════════════════════════════════════════════════

def _invert_f2(M: np.ndarray) -> np.ndarray:
    """Invert an n×n matrix over GF(2) via Gauss-Jordan elimination."""
    n = M.shape[0]
    A = np.hstack([M.astype(np.int8), np.eye(n, dtype=np.int8)]) % 2
    for i in range(n):
        if A[i, i] == 0:
            for j in range(i + 1, n):
                if A[j, i] == 1:
                    A[[i, j]] = A[[j, i]]
                    break
        for j in range(n):
            if j != i and A[j, i] == 1:
                A[j] ^= A[i]
    return A[:, n:] % 2


# ══════════════════════════════════════════════════════════════════════════════
# Vectorised ball / covering helpers
# ══════════════════════════════════════════════════════════════════════════════

def _popcount_u32(a: np.ndarray) -> np.ndarray:
    """Hamming weight of each element in a uint32 array (SWAR algorithm)."""
    a = a.astype(np.uint32, copy=True)
    a -= (a >> np.uint32(1)) & np.uint32(0x55555555)
    a = (a & np.uint32(0x33333333)) + ((a >> np.uint32(2)) & np.uint32(0x33333333))
    a = (a + (a >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return ((a * np.uint32(0x01010101)) >> np.uint32(24)).astype(np.uint32)


def _parity_u32(a: np.ndarray) -> np.ndarray:
    """GF(2) parity (popcount mod 2) of each element in a uint32 array."""
    a = a.astype(np.uint32, copy=True)
    a ^= a >> np.uint32(16)
    a ^= a >> np.uint32(8)
    a ^= a >> np.uint32(4)
    a ^= a >> np.uint32(2)
    a ^= a >> np.uint32(1)
    return (a & np.uint32(1)).astype(np.uint32)


def _encode_vec(v: np.ndarray, n: int) -> int:
    """Big-endian integer encoding of binary vector v (same as vectors_to_ints)."""
    powers = np.int64(1) << np.arange(n - 1, -1, -1, dtype=np.int64)
    return int(np.dot(v.astype(np.int64), powers))


def _col_ints_of(M: np.ndarray, n: int) -> np.ndarray:
    """Big-endian uint32 encoding of each column of binary matrix M (shape n×n).

    col_ints[j] = sum_k  M[k,j] * 2^(n-1-k)

    Matches the vectors_to_ints bit-ordering so that
      parity(x_int XOR c_int, col_ints[j])  ==  dot(x_vec XOR c_vec, M[:,j]) mod 2
    """
    powers = np.uint32(1) << np.arange(n - 1, -1, -1, dtype=np.uint32)   # shape (n,)
    return (M.astype(np.uint32) * powers[:, np.newaxis]).sum(axis=0).astype(np.uint32)


def _compute_S_ints(
    n: int,
    r: int,
    balls: List[Tuple[int, Optional[np.ndarray]]],
    chunk_size: int = 1 << 20,
) -> np.ndarray:
    """Return int64 indices of  F_2^n  minus the union of the given balls.

    Each ball is a tuple (center_int, col_ints) where:
      center_int : big-endian integer encoding of the ball centre.
      col_ints   : shape-(n,) uint32 array of column encodings of basis L,
                   or None for the standard (identity) basis.

    Membership condition:
      x ∈ B_L(c, r)  ⟺  weight( (x_vec ⊕ c_vec) @ L ) ≤ r

    For standard basis (col_ints is None):
      x ∈ B(c, r)   ⟺  popcount(x_int ⊕ c_int) ≤ r

    The universe is partitioned into chunks of `chunk_size` to keep peak
    RAM at O(chunk_size × 4 bytes) per ball pass.
    """
    total = 1 << n
    r_u32 = np.uint32(r)
    # S_mask[i] = True  iff  integer i is NOT in any ball
    S_mask = np.ones(total, dtype=bool)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = np.arange(start, end, dtype=np.uint32)
        in_union = np.zeros(end - start, dtype=bool)

        for center_int, col_ints in balls:
            z = chunk ^ np.uint32(center_int)       # x ⊕ c, element-wise
            if col_ints is None:
                hw = _popcount_u32(z)
            else:
                hw = np.zeros(end - start, dtype=np.uint32)
                for ci in col_ints:                 # one pass per basis column
                    hw += _parity_u32(z & ci)       # GF(2) dot product with col j
            in_union |= (hw <= r_u32)

        S_mask[start:end] &= ~in_union

    return np.flatnonzero(S_mask).astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# Per-trial worker  (top-level function so it is picklable by multiprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def _run_trial(args: Tuple[int, int, int]) -> Dict:
    """Run one (n, r, seed) trial and return per-trial metric dict.

    Three Hamming balls are removed from F_2^n:
      Ball 1 – standard basis,       random centre c1
      Ball 2 – random basis L1,      random centre c2
      Ball 3 – random basis L2,      random centre c3

    Ball membership (all three):
      x ∈ B_L(c, r)  ⟺  weight( (x ⊕ c) @ L ) ≤ r

    Returns
    -------
    dict with keys:  n, r, S_size, SS_size, coverage_density
    """
    n, r, seed = args
    rng = np.random.default_rng(seed)
    universe_size = 1 << n

    # Random centres
    c1 = rng.integers(0, 2, n, dtype=np.int8)
    c2 = rng.integers(0, 2, n, dtype=np.int8)
    c3 = rng.integers(0, 2, n, dtype=np.int8)

    # Random invertible bases
    L1 = generate_random_basis(n, 2, rng=rng)
    L2 = generate_random_basis(n, 2, rng=rng)

    balls = [
        (_encode_vec(c1, n), None),                    # Ball 1: identity basis
        (_encode_vec(c2, n), _col_ints_of(L1, n)),     # Ball 2: basis L1
        (_encode_vec(c3, n), _col_ints_of(L2, n)),     # Ball 3: basis L2
    ]

    # Chunk size: ~32 MB of uint32 per pass, capped to universe
    chunk = min(1 << 23, universe_size)

    S_ints = _compute_S_ints(n, r, balls, chunk_size=chunk)
    S_size = int(len(S_ints))

    if S_size == 0:
        return {
            'n': n, 'r': r,
            'S_size': 0, 'SS_size': 0,
            'coverage_density': 0.0,
        }

    # S+S via in-place FWHT (low-memory path)
    SS_ints = compute_sumset_lowmem_from_ints(S_ints, n, p=2)
    del S_ints

    SS_size = int(len(SS_ints))
    return {
        'n': n, 'r': r,
        'S_size': S_size,
        'SS_size': SS_size,
        'coverage_density': SS_size / universe_size,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Worker-count cap
# ══════════════════════════════════════════════════════════════════════════════

def _worker_cap(n: int, requested: Optional[int]) -> int:
    """Conservative worker count to prevent OOM for large n.

    Approximate peak RAM per worker:
      S_mask (bool, 2^n B) + S_ints (|S|×8 B) + FWHT (2^n×8 B = float64)
      ≈ 9 × 2^n bytes in the worst case.
    """
    hw = cpu_count() or 1
    caps: Dict[int, int] = {24: 4, 25: 2, 26: 1}
    cap = caps.get(n, hw)
    if requested is not None:
        return max(1, min(requested, cap))
    return max(1, min(hw, cap))


# ══════════════════════════════════════════════════════════════════════════════
# Sweep driver
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    configs: List[Tuple[int, List[int]]],
    trials: int,
    seed: int,
    workers: Optional[int],
    label: str,
) -> List[Dict]:
    """Run all (n, r) pairs; print per-trial table; return flat result list."""
    print(f"\n{'═' * 76}")
    print(f"  {label}  [{trials} trial(s) per (n, r)]")
    print(f"{'═' * 76}")
    hdr = (
        f"  {'n':>4}  {'r':>4}  {'trial':>5}  "
        f"{'|S|':>14}  {'|S+S|':>14}  {'|S+S|/2^n':>12}"
    )
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))

    flat: List[Dict] = []
    rng_top = np.random.default_rng(seed)

    for n, r_vals in configs:
        nw = _worker_cap(n, workers)
        print(f"\n  n={n}   |F₂ⁿ| = {1 << n:,}   workers = {nw}")

        for r in sorted(r_vals):
            ball_sz = sum(comb(n, k) for k in range(r + 1))
            seeds = rng_top.integers(0, 1 << 31, size=trials)
            tasks = [(n, r, int(s)) for s in seeds]

            t0 = time.perf_counter()
            if nw > 1:
                with Pool(processes=nw) as pool:
                    results = pool.map(_run_trial, tasks)
            else:
                results = [_run_trial(t) for t in tasks]
            elapsed = time.perf_counter() - t0

            for i, res in enumerate(results):
                print(
                    f"  {n:>4}  {r:>4}  {i:>5}  "
                    f"{res['S_size']:>14,}  {res['SS_size']:>14,}  "
                    f"{res['coverage_density']:>12.6f}"
                )

            flat.extend(results)
            mean_d = float(np.mean([x['coverage_density'] for x in results]))
            n_full = sum(1 for x in results if x['SS_size'] == (1 << n))
            print(
                f"    ↳ ball≈{ball_sz:,}  {elapsed:.1f}s  "
                f"mean_density={mean_d:.4f}  "
                f"full_coverage={n_full}/{trials}"
            )

    return flat


# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse(results: List[Dict]) -> None:
    """Print two analysis sections to stdout."""

    # Aggregate per (n, r)
    by_nr: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for res in results:
        by_nr[(res['n'], res['r'])].append(res['coverage_density'])
    mean_d: Dict[Tuple[int, int], float] = {
        k: float(np.mean(v)) for k, v in by_nr.items()
    }

    print(f"\n{'═' * 76}")
    print("  Analysis")
    print(f"{'═' * 76}")

    # ── [1] min(n) where |S| ≥ 2^(n/2) but coverage fails ────────────────
    print(
        "\n  [1]  Trials where |S| ≥ 2^(n/2)  but  |S+S| < 2^n  (coverage fails)"
    )
    failures = [
        res for res in results
        if res['S_size'] > 0
        and res['S_size'] >= (1 << res['n']) ** 0.5
        and res['SS_size'] < (1 << res['n'])
    ]
    if failures:
        min_n_fail = min(f['n'] for f in failures)
        print(f"       min(n) with such a failure = {min_n_fail}")
        print(
            f"       {'n':>4}  {'r':>4}  "
            f"{'|S|':>14}  {'|S+S|':>14}  {'density':>10}"
        )
        for f in sorted(failures, key=lambda x: (x['n'], x['r'], -x['S_size'])):
            print(
                f"       {f['n']:>4}  {f['r']:>4}  "
                f"{f['S_size']:>14,}  {f['SS_size']:>14,}  "
                f"{f['coverage_density']:>10.6f}"
            )
    else:
        print("       None found across all trials.")
        print(
            "       Every trial with |S| ≥ √(2^n) achieved |S+S| = 2^n."
        )

    # ── [2] derivative d(mean_density)/dr ────────────────────────────────
    print(f"\n  [2]  d(mean_density)/dr at consecutive r values")
    print(
        f"       {'n':>4}  {'r₀':>4}  {'r₁':>4}  "
        f"{'density(r₀)':>13}  {'density(r₁)':>13}  {'Δd/Δr':>10}"
    )

    by_n: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (n, r), d in mean_d.items():
        by_n[n].append((r, d))

    steep: List[Tuple[float, int, int, int]] = []    # (|deriv|, n, r0, r1)
    for n in sorted(by_n):
        pairs = sorted(by_n[n])                       # ascending r
        for i in range(len(pairs) - 1):
            r0, d0 = pairs[i]
            r1, d1 = pairs[i + 1]
            delta_r = r1 - r0
            deriv = (d1 - d0) / delta_r
            print(
                f"       {n:>4}  {r0:>4}  {r1:>4}  "
                f"{d0:>13.6f}  {d1:>13.6f}  {deriv:>+10.6f}"
            )
            steep.append((abs(deriv), n, r0, r1))
        if len(pairs) > 1:
            print()

    if steep:
        steep.sort(reverse=True)
        _, n_b, r0_b, r1_b = steep[0]
        dv_b = (mean_d[(n_b, r1_b)] - mean_d[(n_b, r0_b)]) / (r1_b - r0_b)
        print(
            f"       Steepest boundary:  n={n_b},  "
            f"r: {r0_b}→{r1_b},  Δd/Δr = {dv_b:+.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def make_heatmap(results: List[Dict], output_path: Path) -> None:
    """2-D heatmap: x = r, y = n, colour = mean coverage density.

    Colour scale:  0.0 → red,   0.5 → white,   1.0 → blue.
    Grey cells were not tested (NaN).
    """
    by_nr: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for res in results:
        by_nr[(res['n'], res['r'])].append(res['coverage_density'])

    all_n = sorted({res['n'] for res in results})
    all_r = sorted({res['r'] for res in results})
    n_idx = {n: i for i, n in enumerate(all_n)}
    r_idx = {r: i for i, r in enumerate(all_r)}

    grid = np.full((len(all_n), len(all_r)), np.nan)
    for (n, r), vals in by_nr.items():
        grid[n_idx[n], r_idx[r]] = float(np.mean(vals))

    figw = max(11, len(all_r) * 1.0 + 3)
    figh = max(6,  len(all_n) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(figw, figh))

    # Grey background visible through NaN (masked) cells
    ax.set_facecolor('#b0b0b0')

    im = ax.imshow(
        np.ma.masked_invalid(grid),
        aspect='auto',
        cmap='RdBu',             # red = 0.0,  white = 0.5,  blue = 1.0
        vmin=0.0,
        vmax=1.0,
        origin='lower',          # smallest n at bottom
        interpolation='nearest',
    )

    # Axis labels and ticks
    ax.set_xticks(range(len(all_r)))
    ax.set_xticklabels([str(r) for r in all_r], fontsize=9)
    ax.set_yticks(range(len(all_n)))
    ax.set_yticklabels([str(n) for n in all_n], fontsize=9)
    ax.set_xlabel('r  (ball radius)', fontsize=12)
    ax.set_ylabel('n  (dimension)', fontsize=12)
    ax.set_title(
        'Mean Coverage Density  $|S+S|\\,/\\,2^n$\n'
        '(3 random Hamming balls of radius $r$ removed from $\\mathbb{F}_2^n$)',
        fontsize=12,
    )

    # Minor grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(all_r)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(all_n)), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Annotate each tested cell with its mean density value
    for i in range(len(all_n)):
        for j in range(len(all_r)):
            val = grid[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val - 0.5) > 0.33 else 'black'
                ax.text(
                    j, i, f'{val:.3f}',
                    ha='center', va='center',
                    fontsize=7.5, color=text_color, fontweight='bold',
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label(r'$|S{+}S|\,/\,2^n$', fontsize=11)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0.0  (red)', '0.25', '0.50', '0.75', '1.0  (blue)'])

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Heatmap saved  →  {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Coverage density sweep: 3 random Hamming balls removed from F_2^n; '
            'measure |S+S|/2^n where S = F_2^n \\ balls.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--baseline_trials', type=int, default=20,
        help='Trials per (n,r) in baseline (n = 10, 12, 14, 16, 18).',
    )
    parser.add_argument(
        '--big_trials', type=int, default=5,
        help='Trials per (n,r) in big config (n = 20, 22, 24, 26).',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Worker processes (auto-capped for large n to prevent OOM).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Master RNG seed (baseline); big uses seed+1000.',
    )
    parser.add_argument(
        '--output', type=Path, default=Path(__file__).parent,
        help='Directory for saved files.',
    )
    parser.add_argument(
        '--skip_baseline', action='store_true',
        help='Skip baseline (n ≤ 18); run big configs only.',
    )
    parser.add_argument(
        '--skip_big', action='store_true',
        help='Skip big (n ≥ 20); run baseline configs only.',
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict] = []

    if not args.skip_baseline:
        all_results.extend(run_sweep(
            BASELINE_CONFIG,
            trials=args.baseline_trials,
            seed=args.seed,
            workers=args.workers,
            label='BASELINE  (n = 10, 12, 14, 16, 18)',
        ))

    if not args.skip_big:
        all_results.extend(run_sweep(
            BIG_CONFIG,
            trials=args.big_trials,
            seed=args.seed + 1_000,   # independent RNG sequence
            workers=args.workers,
            label='BIG  (n = 20, 22, 24, 26)  [low-memory FWHT path]',
        ))

    if not all_results:
        print('No results collected. Check --skip_baseline / --skip_big flags.')
        return

    analyse(all_results)

    heatmap_path = args.output / 'coverage_density_heatmap.png'
    make_heatmap(all_results, heatmap_path)

    npz_path = args.output / 'coverage_experiment_results.npz'
    np.savez_compressed(
        npz_path,
        n=np.array([r['n'] for r in all_results], dtype=np.int32),
        r=np.array([r['r'] for r in all_results], dtype=np.int32),
        S_size=np.array([r['S_size'] for r in all_results], dtype=np.int64),
        SS_size=np.array([r['SS_size'] for r in all_results], dtype=np.int64),
        coverage_density=np.array([r['coverage_density'] for r in all_results]),
    )
    print(f"  Raw data saved  →  {npz_path}")


if __name__ == '__main__':
    main()
