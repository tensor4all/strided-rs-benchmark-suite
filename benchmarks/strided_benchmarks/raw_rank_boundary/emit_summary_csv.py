import csv, statistics, sys
def load(p):
    d = {}
    for row in csv.DictReader(open(p)):
        key = (row['op'], int(row['rank']), row['kind'], int(row['total']), int(row['fused_rank']))
        d[key] = (float(row['ns_per_elem']), int(row['allocs_per_call']))
    return d
def emit(outdir, reps, path, label):
    bper = [load(f'{outdir}/base_{r}.csv') for r in range(1, reps+1)]
    cper = [load(f'{outdir}/cand_{r}.csv') for r in range(1, reps+1)]
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['experiment','op','rank','kind','total','fused_rank','reps',
                    'base_median_ns_per_elem','cand_median_ns_per_elem','ratio',
                    'base_min','cand_min','base_max','cand_max','base_rel_spread',
                    'base_allocs_per_call','cand_allocs_per_call'])
        for k in bper[0]:
            b = [p[k][0] for p in bper]; c = [p[k][0] for p in cper]
            bmed = statistics.median(b); cmed = statistics.median(c)
            w.writerow([label, k[0], k[1], k[2], k[3], k[4], reps,
                        f'{bmed:.6f}', f'{cmed:.6f}', f'{cmed/bmed:.4f}',
                        f'{min(b):.6f}', f'{min(c):.6f}', f'{max(b):.6f}', f'{max(c):.6f}',
                        f'{(max(b)-min(b))/bmed:.4f}',
                        bper[0][k][1], cper[0][k][1]])
emit('results-final-1t', 15, 'summary-1t-15.csv', '1t')
emit('results-final-4t', 7, 'summary-4t-7.csv', '4t')
emit('results-null-aa', 7, 'summary-null-aa-7.csv', 'null')
print('written')
