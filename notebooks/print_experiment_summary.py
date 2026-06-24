#!/usr/bin/env python3
"""Print summaries for experiment .npz files.
Usage: python notebooks/print_experiment_summary.py [file.npz]
If no file is given, defaults to `k_balls_experiment_results.npz` in the notebooks folder.
"""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description='Print summary for .npz experiment files')
parser.add_argument('npz', nargs='?', default='k_balls_experiment_results.npz', help='Relative path to .npz file in notebooks/')
args = parser.parse_args()

fpath = Path(__file__).parent / args.npz
if not fpath.exists():
    print('File not found:', fpath)
    raise SystemExit(1)

data = np.load(fpath, allow_pickle=True)
keys = list(data.keys())
print('Loaded', fpath)
print('Contained keys:', keys)

# If there's a summary, print concise stats similar to previous behavior
if 'summary' in data:
    summary = data['summary'].item()
    exp_results = data['exp_results'].item() if 'exp_results' in data else None
    zero_holes = data['zero_holes'] if 'zero_holes' in data else []
    print('\nZero-hole count:', len(zero_holes))
    for n in sorted(summary.keys()):
        print('n=', n)
        for K in sorted(summary[n].keys()):
            s = summary[n][K]
            # compute min dim if missing
            min_dim = s.get('max_subspace_dim_min')
            if min_dim is None and exp_results is not None:
                records = exp_results.get(n, {}).get(K, [])
                dims = [r['max_subspace_dim'] for r in records if r['max_subspace_dim'] is not None]
                min_dim = int(min(dims)) if len(dims) > 0 else None
            print(f" K={K}: trials={s['trials_run']}, zeros={s['S_size_zero_count']}, S_mean={s['S_size_mean']}, subspace_dim_mean={s['max_subspace_dim_mean']}, subspace_dim_min={min_dim}")
    if len(zero_holes) > 0:
        print('\nExamples of zero-hole records (up to 5):')
        for rec in zero_holes[:5]:
            print(rec)

# If duplicates file, print duplicate summaries
# Accept both files that saved the duplicate info and the aggregated structure
if any(k in data for k in ('per_record_duplicates', 'aggregated_dup_counts', 'aggregated_center_counts', 'per_record_duplicates')):
    print('\nDuplicate data present:')
    if 'per_record_duplicates' in data:
        pr = data['per_record_duplicates']
        try:
            pr_list = pr.tolist()
        except Exception:
            pr_list = pr
        print(' Per-record duplicates count:', len(pr_list))
        for rec in pr_list:
            print(rec)
    if 'aggregated_dup_counts' in data:
        ad = data['aggregated_dup_counts']
        try:
            ad_dict = ad.item() if hasattr(ad, 'item') else dict(ad)
        except Exception:
            ad_dict = dict(ad)
        print('\n Aggregated duplicate multiplicities:')
        for c, cnt in ad_dict.items():
            print(f'  {c}: {cnt}')
    if 'aggregated_center_counts' in data:
        ac = data['aggregated_center_counts']
        try:
            ac_dict = ac.item() if hasattr(ac, 'item') else dict(ac)
        except Exception:
            ac_dict = dict(ac)
        print('\n Aggregated total center occurrences:')
        for c, cnt in ac_dict.items():
            print(f'  {c}: {cnt}')

# If low_affine_records present, print a short list
if 'low_affine_records' in data:
    lar = data['low_affine_records']
    try:
        lar_list = lar.tolist()
    except Exception:
        lar_list = lar
    print('\nLow-affine records (count={}):'.format(len(lar_list)))
    for rec in lar_list[:10]:
        print(rec)
