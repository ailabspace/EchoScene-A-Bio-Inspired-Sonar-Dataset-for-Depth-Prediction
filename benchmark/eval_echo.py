import argparse, ast, json, os, sys
import h5py, numpy as np, torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'third_party', 'Visual2Echo'), HERE]
from data_loader.biosonar_dataset import BiosonarDataset
from models.models import ModelBuilder
from options.train_options import TrainOptions
from registration import ColourToDepthWarp
from metrics import metrics, mean_metrics


class EvalDataset(BiosonarDataset):
    def _crop_window(self, fpath, a1, a2):
        a1, a2 = super()._crop_window(fpath, a1, a2)
        n = int(round(float(self.opt.audio_length) * self._geom[fpath][0]))
        if a1.shape[0] < n:
            a1, a2 = (np.pad(a, (0, n - a.shape[0])) for a in (a1, a2))
        return a1, a2


def load_opt(run_dir):
    o = TrainOptions(); o.initialize()
    opt = o.parser.parse_args([])
    for ln in open(os.path.join(run_dir, 'opt.txt')):
        if ': ' not in ln:
            continue
        k, v = ln.rstrip('\n').split(': ', 1)
        try:
            v = ast.literal_eval(v)
        except Exception:
            pass
        setattr(opt, k.strip(), v)
    if isinstance(getattr(opt, 'depth_target_hw', ''), tuple):
        opt.depth_target_hw = ','.join(map(str, opt.depth_target_hw))
    return opt


def fov_masks(ds, hw):
    out = {}
    for f in sorted({p for p, _ in ds.index}):
        rgb = f[:-3] + '_rgb.h5'
        if not os.path.isfile(rgb):
            return None
        with h5py.File(rgb, 'r') as h:
            out[f] = ColourToDepthWarp(*h['rgb'].shape[1:3]).grid(hw)[1].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--weights', default='audiodepth_biosonar_bestval.pth')
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--cap', type=float, default=8.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    opt = load_opt(a.run_dir)
    opt.img_path = opt.audio_path = a.data
    opt.biosonar_rgb_dir = ''
    opt.mode, opt.val_include_test = a.split, False
    opt.use_specaugment, opt.no_audio_augment = False, True

    net = ModelBuilder().build_audiodepth(
        audio_shape=opt.audio_shape, backbone=opt.backbone, mode='mat',
        weights=os.path.join(a.run_dir, a.weights), max_depth=opt.max_depth,
        decoder_spatial_entry=opt.decoder_spatial_entry, audio_norm_type=opt.audio_norm_type,
        audio_norm_groups=opt.audio_norm_groups, mat_nclass=int(getattr(opt, 'mat_nclass', 23))).cuda().eval()
    ds = EvalDataset(); ds.initialize(opt)
    fov = fov_masks(ds, ds.target_hw)

    per, per_fov, i0 = [], [], 0
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, shuffle=False, num_workers=4):
            out = net(b['audio'].cuda().float())
            out = out[0] if isinstance(out, (tuple, list)) else out
            p = (out.float() * opt.max_depth).clamp(0, opt.max_depth)[:, 0].cpu().numpy()
            g = b['depth'].float().reshape(len(p), *ds.target_hw).numpy()
            mk = b['depth_mask'].bool().reshape(len(p), *ds.target_hw).numpy()
            for k in range(len(p)):
                m = mk[k] & (g[k] > 0) & (g[k] <= a.cap)
                if m.sum() >= 10:
                    per.append(metrics(g[k][m], p[k][m]))
                if fov is not None:
                    m = m & fov[ds.index[i0 + k][0]]
                    if m.sum() >= 10:
                        per_fov.append(metrics(g[k][m], p[k][m]))
            i0 += len(p)

    r = {'model': os.path.basename(os.path.dirname(os.path.realpath(a.run_dir))), 'weights': a.weights,
         'data': os.path.basename(os.path.normpath(a.data)), 'split': a.split,
         'full': mean_metrics(per), 'fov': mean_metrics(per_fov) if per_fov else None}
    print(json.dumps(r))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(r, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
