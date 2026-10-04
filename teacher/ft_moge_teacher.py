import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--model_id', default='Ruicheng/moge-2-vits-normal')
p.add_argument('--num_tokens', type=int, default=256)
p.add_argument('--out', required=True)
p.add_argument('--epochs', type=int, default=6)
p.add_argument('--bs', type=int, default=16)
p.add_argument('--lr_enc', type=float, default=5e-6)
p.add_argument('--lr_head', type=float, default=5e-5)
p.add_argument('--wd', type=float, default=1e-4)
p.add_argument('--silog_lambda', type=float, default=0.5)
p.add_argument('--workers', type=int, default=3)
p.add_argument('--seed', type=int, default=1)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'third_party', 'Visual2Echo'))
sys.argv = [sys.argv[0], '--dataset', 'biosonar', '--img_path', args.data, '--audio_path', args.data,
            '--biosonar_rgb_dir', args.data, '--batvision_img_size', '128', '--depth_resize_method', 'nearest',
            '--max_depth', '8', '--exp_name', 'ft_moge_teacher']
from options.train_options import TrainOptions
opt = TrainOptions().parse()
from data_loader import biosonar_dataset as bd
from models.moge_teacher import _ensure_moge_importable
_ensure_moge_importable()
from moge.model.v2 import MoGeModel
from moge.utils.geometry_torch import recover_focal_shift
from huggingface_hub import hf_hub_download


class RGBDepth(bd.BiosonarDataset):
    def __getitem__(self, index):
        fpath, i = self.index[index]
        d = np.asarray(self._handle(fpath)['depth'][i], dtype=np.float32) / 1000.0
        while d.ndim > 2:
            d = d[0]
        valid = np.isfinite(d) & (d > 0) & (d <= float(self.opt.max_depth))
        depth, _ = bd._resize_depth(np.where(valid, d, 0.0).astype(np.float32), valid.astype(np.float32),
                                    self.target_hw, method=self.depth_resize_method)
        return {'img': self._load_rgb(fpath, i), 'depth': torch.as_tensor(depth).float().reshape(1, *self.target_hw)}


def make(mode):
    import copy
    o = copy.copy(opt); o.mode = mode
    ds = RGBDepth(); ds.initialize(o)
    return ds


tr, va = make('train'), make('val')
print(f'train {len(tr)} val {len(va)}', flush=True)
g = torch.Generator(); g.manual_seed(args.seed)
seed_worker = lambda wid: (np.random.seed(args.seed * 100 + wid), random.seed(args.seed * 100 + wid))
dl_tr = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                                    generator=g, worker_init_fn=seed_worker, persistent_workers=True)
dl_va = torch.utils.data.DataLoader(va, 32, shuffle=False, num_workers=args.workers)

dev = 'cuda'
ck_path = args.model_id if os.path.exists(args.model_id) else hf_hub_download(args.model_id, 'model.pt')
ck = torch.load(ck_path, map_location='cpu', weights_only=True)
model = MoGeModel(**ck['model_config']); model.load_state_dict(ck['model'], strict=False)
model = model.to(dev)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
MAXD = float(opt.max_depth)


def metric_depth(net, x_norm):
    x = x_norm * STD + MEAN
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = net(x, num_tokens=args.num_tokens)
    pts = out['points'].float()
    with torch.no_grad():
        m = out['mask'].float() > 0.5 if 'mask' in out else None
        _, shift = recover_focal_shift(pts.detach(), m)
    z = (pts[..., 2] + shift[:, None, None]) * out['metric_scale'].float()[:, None, None]
    return F.interpolate(z[:, None], size=tuple(tr.target_hw), mode='bilinear', align_corners=False)


def silog(pred, gt, m, lam):
    lp, lg = torch.log(pred.clamp_min(1e-3)), torch.log(gt.clamp_min(1e-3))
    loss = 0.
    for b in range(pred.shape[0]):
        if m[b].sum() < 50:
            continue
        d = (lp[b] - lg[b])[m[b]]
        loss = loss + torch.sqrt((d ** 2).mean() - lam * d.mean() ** 2 + 1e-8)
    return loss / pred.shape[0]


@torch.no_grad()
def evaluate():
    model.eval()
    acc = {k: [] for k in ('rmse', 'absrel', 'd1', 'rmse_al', 'absrel_al', 'd1_al')}
    for batch in dl_va:
        x, gt = batch['img'].to(dev), batch['depth'].to(dev)
        pr = metric_depth(model, x).clamp(0, MAXD)
        for b in range(x.shape[0]):
            v = (gt[b] > 0) & (gt[b] < MAXD) & (pr[b] > 1e-3)
            if v.sum() < 50:
                continue
            gv, pv = gt[b][v], pr[b][v].clamp_min(1e-3)
            for q, s in ((pv, ''), (pv * gv.median() / pv.median(), '_al')):
                e = q - gv
                acc['rmse' + s].append(e.pow(2).mean().sqrt().item())
                acc['absrel' + s].append((e.abs() / gv).mean().item())
                acc['d1' + s].append((torch.maximum(q / gv, gv / q) < 1.25).float().mean().item())
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


enc = [q for n, q in model.named_parameters() if n.startswith('encoder.')]
rest = [q for n, q in model.named_parameters() if not n.startswith('encoder.')]
optim = torch.optim.AdamW([{'params': enc, 'lr': args.lr_enc}, {'params': rest, 'lr': args.lr_head}],
                          weight_decay=args.wd)
total = args.epochs * len(dl_tr)
sched = torch.optim.lr_scheduler.LambdaLR(
    optim, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
r0 = evaluate()
print('zero-shot val', json.dumps({k: round(v, 4) for k, v in r0.items()}), flush=True)
best, step, t0 = r0['rmse'], 0, time.time()
model.train()
for ep in range(args.epochs):
    for batch in dl_tr:
        x, gt = batch['img'].to(dev), batch['depth'].to(dev)
        flip = random.random() < 0.5
        pr = metric_depth(model, x.flip(-1) if flip else x)
        pr = pr.flip(-1) if flip else pr
        loss = silog(pr, gt, (gt > 0) & (gt < MAXD), args.silog_lambda)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); sched.step(); step += 1
        if step % 100 == 0:
            print(f'ep {ep} step {step}/{total} loss {loss.item():.4f} {(time.time() - t0) / step:.2f}s/it', flush=True)
    r = evaluate()
    tag = ''
    if r['rmse'] < best:
        best = r['rmse']; tag = ' *best'
        torch.save({'model_config': ck['model_config'], 'model': model.state_dict(),
                    'ft_args': vars(args), 'step': step, 'val': r}, args.out)
    print(f'VAL ep {ep} step {step}', json.dumps({k: round(v, 4) for k, v in r.items()}) + tag, flush=True)
print('done best val rmse', best, flush=True)
