import numpy as np


def metrics(gt, pred):
    pred = np.clip(pred, 1e-3, None)
    th = np.maximum(gt / pred, pred / gt)
    return dict(abs_rel=np.mean(np.abs(gt - pred) / gt), sq_rel=np.mean((gt - pred) ** 2 / gt),
                rmse=np.sqrt(np.mean((gt - pred) ** 2)), log10=np.mean(np.abs(np.log10(pred) - np.log10(gt))),
                mae=np.mean(np.abs(gt - pred)), d1=(th < 1.25).mean(), d2=(th < 1.25 ** 2).mean(),
                d3=(th < 1.25 ** 3).mean())


def mean_metrics(per):
    r = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    r['n'] = len(per)
    return r
