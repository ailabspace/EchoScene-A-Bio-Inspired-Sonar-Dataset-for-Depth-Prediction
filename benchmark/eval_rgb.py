import argparse, json, os, sys
from types import SimpleNamespace
import h5py, numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'third_party', 'Visual2Echo'), HERE]
from data_loader import biosonar_dataset as bd
from registration import ColourToDepthWarp
from metrics import metrics, mean_metrics


class RGBWindows(bd.BiosonarDataset):
    def __getitem__(self, index):
        fpath, i = self.index[index]
        d = np.asarray(self._handle(fpath)['depth'][i], dtype=np.float32) / 1000.0
        while d.ndim > 2:
            d = d[0]
        valid = np.isfinite(d) & (d > 0) & (d <= float(self.opt.max_depth))
        depth, mask = bd._resize_depth(np.where(valid, d, 0.0).astype(np.float32), valid.astype(np.float32),
                                       self.target_hw, method='nearest')
        rgb_path = fpath[:-3] + '_rgb.h5'
        if rgb_path not in self._rgb_handles:
            self._rgb_handles[rgb_path] = h5py.File(rgb_path, 'r')
        rgb = np.asarray(self._rgb_handles[rgb_path]['rgb'][i])
        return {'rgb': torch.from_numpy(rgb), 'depth': torch.as_tensor(depth).float().reshape(self.target_hw),
                'mask': torch.as_tensor(mask).bool().reshape(self.target_hw), 'pos': index}


def make_dataset(data, split, max_depth):
    opt = SimpleNamespace(img_path=data, audio_path=data, batvision_img_size=128, depth_target_hw='128,128',
                          depth_resize_method='nearest', biosonar_rgb_dir='', teacher_img_size=224,
                          audio_sampling_rate=320000, audio_length=0.075, audio_crop_pre_roll=0.002,
                          audio_nfft=512, audio_bandpass_lo=0, audio_bandpass_hi=0, audio_butter_order=0,
                          mode=split, val_include_test=False, max_depth=max_depth)
    ds = RGBWindows(); ds.initialize(opt)
    return ds


class MoGe:
    def __init__(self, model_id, num_tokens, square):
        from moge.model.v2 import MoGeModel
        self.net = MoGeModel.from_pretrained(model_id).cuda().eval()
        self.num_tokens, self.square = num_tokens or None, square

    @torch.no_grad()
    def __call__(self, rgb):
        x = rgb.cuda().permute(0, 3, 1, 2).float() / 255.0
        h, w = x.shape[-2:]
        if self.square:
            x = F.interpolate(x, size=(self.square, self.square), mode='bilinear', align_corners=False)
        out = []
        for xi in x:
            d = self.net.infer(xi, num_tokens=self.num_tokens, apply_mask=False)['depth']
            out.append(F.interpolate(d[None, None].float(), size=(h, w), mode='bilinear', align_corners=False)[0, 0])
        return torch.stack(out)


class DA3:
    def __init__(self, model_id):
        from depth_anything_3.api import DepthAnything3
        self.net = DepthAnything3.from_pretrained(model_id).cuda().eval()

    @torch.no_grad()
    def __call__(self, rgb):
        h, w = rgb.shape[1:3]
        d = torch.as_tensor(np.asarray(self.net.inference(list(rgb.numpy())).depth)).float().cuda()
        return F.interpolate(d[:, None], size=(h, w), mode='bilinear', align_corners=False)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['moge2', 'da3'], required=True)
    ap.add_argument('--model_id', default='')
    ap.add_argument('--num_tokens', type=int, default=0)
    ap.add_argument('--square', type=int, default=0)
    ap.add_argument('--name', default='')
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--cap', type=float, default=8.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    if a.method == 'moge2':
        model = MoGe(a.model_id or 'Ruicheng/moge-2-vitl-normal', a.num_tokens, a.square)
    else:
        model = DA3(a.model_id or 'depth-anything/DA3NESTED-GIANT-LARGE')
    ds = make_dataset(a.data, a.split, a.cap)
    warps, acc = {}, {'raw': [], 'median': [], 'affine': []}
    for s in torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=4):
        pred = torch.nan_to_num(model(s['rgb'][None])[0], nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 100)
        key = tuple(s['rgb'].shape[:2])
        if key not in warps:
            warps[key] = ColourToDepthWarp(*key)
        g, inside = warps[key].grid(ds.target_hw, z_colour=pred)
        p = ColourToDepthWarp.apply(pred[None], g, inside, mode='nearest')[0].cpu().numpy()
        gt = s['depth'].numpy()
        m = s['mask'].numpy() & inside.cpu().numpy() & (gt > 0) & (gt <= a.cap) & (p > 1e-3)
        if m.sum() >= 10:
            pv, gv = p[m].astype(np.float64), gt[m].astype(np.float64)
            A = np.stack([pv, np.ones_like(pv)], 1)
            for proto, q in (('raw', pv), ('median', pv * np.median(gv) / np.median(pv)),
                             ('affine', A @ np.linalg.lstsq(A, gv, rcond=None)[0])):
                acc[proto].append(metrics(gv, np.clip(q, 1e-3, a.cap)))

    r = {'model': a.name or a.method, 'model_id': model.__class__.__name__ + ':' + (a.model_id or 'default'),
         'data': os.path.basename(os.path.normpath(a.data)), 'split': a.split,
         **{proto: mean_metrics(v) for proto, v in acc.items()}}
    print(json.dumps(r))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(r, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
