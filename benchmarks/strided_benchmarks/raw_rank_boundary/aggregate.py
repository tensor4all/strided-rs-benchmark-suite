import csv, statistics, sys, os
def load(p):
    d = {}
    for row in csv.DictReader(open(p)):
        key = (row['op'], int(row['rank']), row['kind'], int(row['total']), int(row['fused_rank']))
        d[key] = (float(row['ns_per_elem']), int(row['allocs_per_call']))
    return d
outdir, reps = sys.argv[1], int(sys.argv[2])
bper = [load(os.path.join(outdir, f'base_{r}.csv')) for r in range(1, reps+1)]
cper = [load(os.path.join(outdir, f'cand_{r}.csv')) for r in range(1, reps+1)]
rows = []
for k in bper[0]:
    b = [p[k][0] for p in bper]; c = [p[k][0] for p in cper]
    bmed, cmed = statistics.median(b), statistics.median(c)
    rows.append(dict(k=k, bmed=bmed, cmed=cmed, bmin=min(b), cmin=min(c), bmax=max(b), cmax=max(c),
                     spread=(max(b)-min(b))/bmed, bal=bper[0][k][1], cal=cper[0][k][1]))
print(f"{'op':<18}{'rank':>4}{'kind':>18}{'tot':>8}{'fr':>3}{'base_med':>9}{'cand_med':>9}{'ratio':>7}{'bmin':>8}{'cmin':>8}{'bspread':>8}{'bal':>4}{'cal':>4}")
for r in rows:
    k = r['k']
    print(f"{k[0]:<18}{k[1]:>4}{k[2]:>18}{k[3]:>8}{k[4]:>3}{r['bmed']:>9.4f}{r['cmed']:>9.4f}{r['cmed']/r['bmed']:>7.3f}{r['bmin']:>8.4f}{r['cmin']:>8.4f}{r['spread']:>8.2f}{r['bal']:>4}{r['cal']:>4}")
def ratio(r): return r['cmed']/r['bmed']
common = [r for r in rows if r['k'][1] <= 8]
cliff = [r for r in rows if r['k'][1] > 8 and r['k'][4] > 8]
small = [r for r in rows if r['k'][1] > 8 and r['k'][4] <= 8]
print()
print(f"COMMON PATH (original rank <= 8): {len(common)} cases")
clean = [r for r in common if r['spread'] <= 0.05]; noisy = [r for r in common if r['spread'] > 0.05]
print(f"  clean (baseline spread <= 5%): {len(clean)}, worst median ratio {max(ratio(r) for r in clean):.3f}, gate 1.05 -> {'PASS' if max(ratio(r) for r in clean) <= 1.05 else 'FAIL'}")
for r in sorted(clean, key=ratio, reverse=True)[:3]:
    print(f"    worst: rank {r['k'][1]} {r['k'][2]} {r['k'][0]} ratio={ratio(r):.3f} spread={r['spread']:.2f}")
print(f"  low-confidence (baseline spread > 5%): {len(noisy)}, worst ratio {max(ratio(r) for r in noisy):.3f}")
for r in sorted(noisy, key=ratio, reverse=True)[:4]:
    print(f"    rank {r['k'][1]:>2} {r['k'][2]:>18} {r['k'][0]:<18} ratio={ratio(r):.3f} spread={r['spread']:.2f} cand_med<=base_max={r['cmed']<=r['bmax']}")
print(f"CLIFF CASES (original rank > 8 and fused_rank > 8): {len(cliff)}")
print(f"  worst (max) ratio {max(ratio(r) for r in cliff):.3f} (gate <= {1/1.3:.3f}), best {min(ratio(r) for r in cliff):.3f} -> {'PASS' if max(ratio(r) for r in cliff) <= 1/1.3 else 'FAIL'}")
print(f"  baseline allocs/call {sorted(set(r['bal'] for r in cliff))} -> candidate {sorted(set(r['cal'] for r in cliff))}")
print(f"  speedups: min {1/max(ratio(r) for r in cliff):.2f}x max {1/min(ratio(r) for r in cliff):.2f}x")
print(f"FUSED-SMALL HIGH-RANK (reported, not gated): {len(small)} cases")
for r in sorted(small, key=ratio, reverse=True):
    print(f"    rank {r['k'][1]:>2} {r['k'][2]:>18} {r['k'][0]:<18} base={r['bmed']:.4f} cand={r['cmed']:.4f} ratio={ratio(r):.3f} spread={r['spread']:.2f} cal={r['cal']}")
