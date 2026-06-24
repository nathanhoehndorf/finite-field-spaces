import numpy as np
from pathlib import Path

def gf2_rank(mat):
    # mat is 2D numpy array of ints 0/1 shape (m,n)
    A = mat.copy() % 2
    m, n = A.shape
    rank = 0
    row = 0
    for col in range(n):
        # find pivot in rows row..m-1 with A[i,col]==1
        piv = None
        for i in range(row, m):
            if A[i, col] == 1:
                piv = i
                break
        if piv is None:
            continue
        # swap rows
        if piv != row:
            A[[piv, row]] = A[[row, piv]]
        # eliminate below
        for i in range(row+1, m):
            if A[i, col] == 1:
                A[i] ^= A[row]
        row += 1
        rank += 1
        if row == m:
            break
    return rank

p = Path(__file__).parent / 'k_balls_experiment_results.npz'
if not p.exists():
    print('Results file not found:', p)
    raise SystemExit(1)

data = np.load(p, allow_pickle=True)
zero_holes = data['zero_holes']
exp_results = data['exp_results'].item() if 'exp_results' in data else None

# Filter zero_holes for K=6
k = 6
kh = [rec for rec in zero_holes if rec.get('K') == k]
print(f'Total zero-hole records: {len(zero_holes)}; K={k} records: {len(kh)}')
if len(kh) == 0:
    print('No K=6 zero-hole records found.')
    raise SystemExit(0)

summary = {'total': len(kh), 'with_duplicates': 0, 'rank_counts': {}, 'affine_rank_counts': {}}

for idx, rec in enumerate(kh):
    centers = np.array(rec['centers'], dtype=int)
    # centers shape should be (6,n)
    if centers.ndim != 2:
        print('Unexpected centers format in rec', rec)
        continue
    m, n = centers.shape
    # duplicates
    unique_centers = {tuple(row) for row in centers}
    dup = (len(unique_centers) < m)
    if dup:
        summary['with_duplicates'] += 1
    # GF(2) rank of centers as matrix (linear rank)
    rank = gf2_rank(centers)
    summary['rank_counts'][rank] = summary['rank_counts'].get(rank, 0) + 1
    # affine rank: subtract first center and compute rank of differences
    diffs = (centers - centers[0]) % 2
    affine_rank = gf2_rank(diffs[1:]) if m > 1 else 0
    summary['affine_rank_counts'][affine_rank] = summary['affine_rank_counts'].get(affine_rank, 0) + 1
    print(f"Record {idx}: trial={rec.get('trial')} duplicates={dup} linear_rank={rank} affine_rank={affine_rank}")

print('\nSummary:')
print(' Total K=6 zero-hole records:', summary['total'])
print(' Records with duplicate centers:', summary['with_duplicates'])
print(' Linear rank distribution:', summary['rank_counts'])
print(' Affine rank distribution:', summary['affine_rank_counts'])

# Identify records where affine rank is smaller than the full expected affine span
low_affine_records = []
for rec in kh:
    centers = np.array(rec['centers'], dtype=int)
    m, n = centers.shape
    diffs = (centers - centers[0]) % 2
    affine_rank = gf2_rank(diffs[1:]) if m > 1 else 0
    # expected full affine span dimension is at most min(m-1, n)
    if affine_rank < min(m-1, n):
        low_affine_records.append({'trial': rec.get('trial'), 'affine_rank': int(affine_rank), 'centers': centers.tolist()})

# Save summary and detailed low-affine records
out = Path(__file__).parent / f'k6_zero_holes_inspection_summary.npz'
np.savez_compressed(out, summary=summary, low_affine_records=low_affine_records)
print('Saved summary to', out)

from collections import Counter

# Find duplicate centers per record and aggregate counts across records
per_record_duplicates = []
aggregated_center_counts = Counter()
aggregated_dup_counts = Counter()

for rec in kh:
    centers = [tuple(map(int, c)) for c in rec['centers']]
    ctr = Counter(centers)
    # any center with count>1 in this record
    dups = {c:cnt for c,cnt in ctr.items() if cnt > 1}
    if dups:
        per_record_duplicates.append({'trial': rec.get('trial'), 'duplicates': dups})
        for c, cnt in dups.items():
            aggregated_dup_counts[c] += cnt
    # aggregate total occurrences across records
    for c, cnt in ctr.items():
        aggregated_center_counts[c] += cnt

print('\nDuplicate-centers per-record (trial -> {center: multiplicity}):')
for rec in per_record_duplicates:
    print(f" trial={rec['trial']}: {rec['duplicates']}")

print('\nAggregated duplicate multiplicities (center -> total duplicated count across records):')
for c, cnt in aggregated_dup_counts.items():
    print(f" {c}: {cnt}")

print('\nAggregated total center occurrences across all K=6 zero-hole records (center -> count):')
for c, cnt in aggregated_center_counts.items():
    print(f" {c}: {cnt}")

# Save duplicate info
out_dup = Path(__file__).parent / 'k6_zero_holes_duplicates.npz'
np.savez_compressed(out_dup, per_record_duplicates=per_record_duplicates, aggregated_dup_counts=dict(aggregated_dup_counts), aggregated_center_counts=dict(aggregated_center_counts))
print('Saved duplicate info to', out_dup)
