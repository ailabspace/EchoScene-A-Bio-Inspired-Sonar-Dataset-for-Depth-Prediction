import argparse, glob, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, '..', 'splits')
ALIASES = {'block_d_1f_office_1.h5': 'block_d_1f_1.h5', 'block_d_1f_office_2.h5': 'block_d_1f_2.h5',
           'block_d_1f_office_3.h5': 'block_d_1f_3.h5', 'block_d_1f_office_out.h5': 'block_d_1f_4.h5'}


def rgb_name(f):
    return f[:-3] + '_rgb.h5'


def link_view(files, found, out):
    os.makedirs(out, exist_ok=True)
    missing, no_rgb = [], []
    for f in files:
        src = found.get(f) or found.get(ALIASES.get(f, ''))
        if src is None:
            missing.append(f)
            continue
        for s, d in ((src, f), (src[:-3] + '_rgb.h5', rgb_name(f))):
            if not os.path.isfile(s):
                no_rgb.append(f)
                continue
            dst = os.path.join(out, d)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(s), dst)
    return missing, no_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'data'))
    a = ap.parse_args()

    found = {}
    for p in glob.glob(os.path.join(a.release, '**', '*.h5'), recursive=True):
        if not p.endswith('_rgb.h5'):
            found.setdefault(os.path.basename(p), p)

    ok = True
    for name in ('board', 'dark'):
        sp = json.load(open(os.path.join(SPLITS, name, 'splits.json')))
        files = sorted({f for k in ('train', 'val', 'test') for f in sp.get(k, {}).get('files', [])})
        out = os.path.join(a.out, name)
        missing, no_rgb = link_view(files, found, out)
        for j in ('splits.json', 'valid_index.json'):
            if os.path.isfile(os.path.join(SPLITS, name, j)):
                shutil.copy(os.path.join(SPLITS, name, j), os.path.join(out, j))
        print(f'[{name}] {len(files) - len(missing)}/{len(files)} recordings -> {out}')
        if missing:
            ok = False
            print(f'  MISSING recordings: {" ".join(missing)}')
        if no_rgb:
            print(f'  no <rec>_rgb.h5 (needed for the teacher and RGB baselines): {" ".join(no_rgb)}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
