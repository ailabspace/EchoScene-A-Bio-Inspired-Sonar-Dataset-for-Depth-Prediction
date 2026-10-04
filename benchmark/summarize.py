import glob, json, os, sys
from collections import defaultdict

COLS = ['abs_rel', 'rmse', 'log10', 'mae', 'd1', 'd2', 'd3']
rows = defaultdict(list)
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    r = json.load(open(f))
    for proto in ('full', 'fov', 'raw', 'median', 'affine'):
        if r.get(proto):
            region = 'full grid' if proto == 'full' else 'colour FOV'
            rows[(r['data'], r['split'])].append((r['model'], proto if proto not in ('full', 'fov') else 'echo',
                                                  region, r[proto]))
for (data, split), rs in rows.items():
    print(f'\n### {data} / {split}\n')
    print('| model | protocol | pixels | n | ' + ' | '.join(c.upper() for c in COLS) + ' |')
    print('|' + '---|' * (4 + len(COLS)))
    for m, proto, region, v in sorted(rs, key=lambda t: (t[2], t[0], t[1])):
        print(f'| {m} | {proto} | {region} | {v["n"]} | ' + ' | '.join(f'{v[c]:.3f}' for c in COLS) + ' |')
