"""
Command-line interface for ffspaces.

Usage
-----
    ffspaces run [options]          # run the k-balls covering experiment
    ffspaces memory-check [--n N]   # print memory estimates for dimension n
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import numpy as np

def _run_trial_standard(
    n: int,
    k: int,
    radius: int,
    p: int,
    rng: np.random.Generator,
    exhaustive: bool,
) -> dict:
    """Run one trial using the standard (dense) implementations."""
    from ffspaces.core import generate_space, generate_random_basis
    from ffspaces.covers import generate_covering, complement
    from ffspaces.operators import compute_sumset, find_maximum_subspace_dimension

    universe = generate_space(n, p)
    bases = [generate_random_basis(n, p, rng=rng) for _ in range(k)]
    centers = [universe[rng.integers(0, len(universe))] for _ in range(k)]

    # universe=None uses the weight-ball path (same convention as lowmem mode)
    # so that both modes produce identical results for the same seed.
    covered = generate_covering(centers, radius, bases=bases, p=p, universe=None)
    S = complement(universe, covered)
    S_size = len(S)

    max_dim = None
    if S_size > 0:
        S_plus_S = compute_sumset(S, p)
        if len(S_plus_S) == len(universe):
            max_dim = n
        else:
            max_dim = find_maximum_subspace_dimension(
                S_plus_S, p=p, exhaustive=exhaustive
            )

    return {"S_size": S_size, "max_subspace_dim": max_dim}


def _run_trial_lowmem(
    n: int,
    k: int,
    radius: int,
    p: int,
    rng: np.random.Generator,
    exhaustive: bool,
) -> dict:
    """Run one trial using the memory-efficient implementations."""
    from ffspaces.core import generate_random_basis
    from ffspaces.lowmem import (
        generate_covering_ints_lowmem,
        complement_ints_lowmem,
        compute_sumset_lowmem_from_ints,
        find_maximum_subspace_dimension_lowmem,
    )

    if p != 2:
        # Low-memory integer path is only implemented for p=2; fall back.
        return _run_trial_standard(n, k, radius, p, rng, exhaustive)

    N = 1 << n  # universe size (p=2 only)

    # Generate bases and centers in the same RNG call order as _run_trial_standard
    # so that results are directly comparable for the same seed.
    bases = [generate_random_basis(n, p, rng=rng) for _ in range(k)]

    from ffspaces.fwht_operators import ints_to_vectors
    centers = [
        ints_to_vectors(np.array([rng.integers(0, N)]), n, p=2)[0]
        for _ in range(k)
    ]

    covered_ints = generate_covering_ints_lowmem(n, centers, radius, bases=bases, p=p)

    S_ints = complement_ints_lowmem(n, covered_ints, p=p)
    S_size = len(S_ints)

    max_dim = None
    if S_size > 0:
        sumset_ints = compute_sumset_lowmem_from_ints(S_ints, n, p=p)

        if len(sumset_ints) == N:
            max_dim = n
        else:
            max_dim = find_maximum_subspace_dimension_lowmem(sumset_ints, n, p=p)

    del covered_ints, S_ints
    return {"S_size": S_size, "max_subspace_dim": max_dim}


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the k-balls covering experiment."""
    n, k, radius, p = args.n, args.k, args.radius, args.p
    trials, seed = args.trials, args.seed
    low_memory = args.low_memory
    exhaustive = args.exhaustive
    jobs = args.jobs
    output = args.output

    if jobs < 1:
        print("ERROR: --jobs must be at least 1", file=sys.stderr)
        sys.exit(1)

    if not low_memory and n > 20:
        print(
            f"WARNING: n={n} with standard mode may require many GB of RAM. "
            "Consider --low-memory.",
            file=sys.stderr,
        )

    if low_memory:
        from ffspaces.lowmem import estimate_memory_gb
        est = estimate_memory_gb(n, p)
        sumset_gb = est["compute_sumset_lowmem_GB"]
        mask_gb = est["complement_bool_mask_GB"]
        print(
            f"[low-memory mode]  n={n}, p={p}\n"
            f"  Estimated peak per trial: sumset FWHT ~{sumset_gb:.1f} GB, "
            f"complement mask ~{mask_gb:.2f} GB"
        )

    mode_label = "low-memory" if low_memory else "standard"
    run_trial = _run_trial_lowmem if low_memory else _run_trial_standard

    print(
        f"Running: n={n}, k={k}, radius={radius}, p={p}, "
        f"trials={trials}, mode={mode_label}, jobs={jobs}"
    )

    rng = np.random.default_rng(seed)
    results: list[dict] = []
    start = time.time()

    if jobs > 1:
        # Each worker gets its own seed derived from the master RNG so results
        # are reproducible regardless of scheduling order.
        from multiprocessing import Pool
        trial_seeds = rng.integers(0, 2**31, size=trials, dtype=np.int64)

        def _worker(trial_seed: int) -> dict:
            worker_rng = np.random.default_rng(int(trial_seed))
            return run_trial(n, k, radius, p, worker_rng, exhaustive)

        with Pool(processes=jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, trial_seeds)):
                results.append(result)
                if args.progress and (i + 1) % max(1, trials // 20) == 0:
                    elapsed = time.time() - start
                    print(f"  {i+1}/{trials} trials  elapsed={elapsed:.1f}s")
    else:
        for t in range(trials):
            result = run_trial(n, k, radius, p, rng, exhaustive)
            results.append(result)
            if args.progress and (t + 1) % max(1, trials // 20) == 0:
                elapsed = time.time() - start
                print(f"  {t+1}/{trials} trials  elapsed={elapsed:.1f}s")

    elapsed = time.time() - start

    # Summarise
    S_sizes = [r["S_size"] for r in results]
    dims = [r["max_subspace_dim"] for r in results if r["max_subspace_dim"] is not None]
    zero_coverings = sum(1 for s in S_sizes if s == 0)

    print(f"\nCompleted {len(results)} trials in {elapsed:.2f}s")
    print(f"  Perfect coverings (S=∅): {zero_coverings}/{len(results)}")
    if S_sizes:
        print(f"  S size: mean={np.mean(S_sizes):.1f}  median={np.median(S_sizes):.1f}")
    if dims:
        print(
            f"  max_subspace_dim: mean={np.mean(dims):.2f}  "
            f"min={min(dims)}  max={max(dims)}"
        )

    summary = {
        "n": n,
        "k": k,
        "radius": radius,
        "p": p,
        "trials_completed": len(results),
        "mode": mode_label,
        "elapsed_s": elapsed,
        "zero_coverings": zero_coverings,
        "S_size_mean": float(np.mean(S_sizes)) if S_sizes else None,
        "S_size_median": float(np.median(S_sizes)) if S_sizes else None,
        "max_subspace_dim_mean": float(np.mean(dims)) if dims else None,
        "max_subspace_dim_min": int(min(dims)) if dims else None,
        "max_subspace_dim_max": int(max(dims)) if dims else None,
    }

    if output:
        import pathlib
        path = pathlib.Path(output)
        if path.suffix == ".npz":
            np.savez_compressed(path, results=results, summary=summary)
        else:
            path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"Results saved to {path}")


def _cmd_memory_check(args: argparse.Namespace) -> None:
    """Print estimated memory requirements for dimension n."""
    from ffspaces.lowmem import estimate_memory_gb

    for n in args.n:
        print(f"\n── n={n}, p={args.p} ──")
        est = estimate_memory_gb(n, args.p)
        rows = [
            ("generate_space (standard)",     est["generate_space_standard_GB"]),
            ("compute_sumset (standard FFT)",  est["compute_sumset_standard_GB"]),
            ("compute_sumset (low-memory FWHT)", est["compute_sumset_lowmem_GB"]),
            ("complement bool mask (low-mem)", est["complement_bool_mask_GB"]),
            ("S integer array (worst case)",   est["S_ints_worst_case_GB"]),
            ("sumset integer array (worst case)", est["sumset_ints_worst_case_GB"]),
        ]
        width = max(len(label) for label, _ in rows)
        for label, gb in rows:
            bar = "█" * min(40, max(1, int(gb * 2)))
            flag = "  ← low-mem" if "low-mem" in label.lower() else ""
            print(f"  {label:<{width}}  {gb:>7.2f} GB  {bar}{flag}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffspaces",
        description=(
            "Finite-field vector space experiments.\n\n"
            "Use --low-memory on 'run' for n>20 to avoid out-of-memory crashes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run the k-balls covering experiment.",
        description=(
            "Generates K random Hamming balls in F_p^n, computes the complement S,\n"
            "and finds the maximum linear subspace dimension in S+S."
        ),
    )
    run_p.add_argument("--n", type=int, default=8, help="Dimension n (default: 8)")
    run_p.add_argument("--k", type=int, default=3, help="Number of balls K (default: 3)")
    run_p.add_argument("--radius", type=int, default=1, help="Ball radius (default: 1)")
    run_p.add_argument("--p", type=int, default=2, help="Field characteristic p (default: 2)")
    run_p.add_argument("--trials", type=int, default=100, help="Number of trials (default: 100)")
    run_p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    run_p.add_argument("--jobs", type=int, default=1, help="Parallel workers (default: 1)")
    run_p.add_argument(
        "--low-memory",
        action="store_true",
        default=False,
        help=(
            "Use memory-efficient implementations (required for n>20 on most machines).\n"
            "For p=2: real float64 FWHT instead of complex128 FFT,\n"
            "combinatorial ball generation (no universe array),\n"
            "and boolean-mask complement."
        ),
    )
    run_p.add_argument(
        "--exhaustive",
        action="store_true",
        default=False,
        help="Use exhaustive subspace search (slow; ignored in low-memory mode).",
    )
    run_p.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=True,
        help="Suppress per-trial progress output.",
    )
    run_p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Save results to this path.  .npz for compressed numpy, "
            "any other extension for JSON."
        ),
    )

    mc_p = sub.add_parser(
        "memory-check",
        help="Print estimated peak memory for dimension n.",
        description=(
            "Prints estimated peak memory for both standard and low-memory mode\n"
            "at the given dimension(s)."
        ),
    )
    mc_p.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[20, 24, 28, 30],
        help="Dimension(s) to estimate (default: 20 24 28 30)",
    )
    mc_p.add_argument("--p", type=int, default=2, help="Field characteristic p (default: 2)")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point registered in pyproject.toml as ``ffspaces``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "memory-check":
        _cmd_memory_check(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
