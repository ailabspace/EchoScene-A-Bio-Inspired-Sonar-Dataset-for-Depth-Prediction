import numpy as np
import torch
import torch.nn.functional as F

K_DEPTH = (422.04, 422.04, 425.98, 238.91, 848, 480)
K_COLOUR = {'1280x720': (917.79, 917.75, 651.47, 336.95, 1280, 720),
            '640x480': (611.9, 611.8, 327.6, 224.6, 640, 480)}


def colour_mode(h, w):
    return '1280x720' if h / w < 0.65 else '640x480'


def _extrinsics():
    q = np.array([0.0080, -0.0003, 0.0020, 1.0])
    qx, qy, qz, qw = q / np.linalg.norm(q)
    Rb = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                   [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                   [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    tb = np.array([-0.0003, 0.0146, 0.0001])
    C = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    return C @ Rb.T @ C.T, -C @ Rb.T @ tb


R_DC, T_DC = _extrinsics()


class ColourToDepthWarp:
    def __init__(self, rgb_h, rgb_w):
        self.kc = K_COLOUR[colour_mode(rgb_h, rgb_w)]
        self._rays = {}

    def _grid_rays(self, hw, device='cpu'):
        key = (hw, str(device))
        if key not in self._rays:
            H, W = hw
            fx, fy, cx, cy, Wd, Hd = K_DEPTH
            jj, ii = np.meshgrid(np.arange(W), np.arange(H))
            ud, vd = (jj + 0.5) * Wd / W - 0.5, (ii + 0.5) * Hd / H - 0.5
            r = np.stack([(ud - cx) / fx, (vd - cy) / fy, np.ones_like(ud, dtype=float)], -1)
            self._rays[key] = torch.from_numpy(r @ R_DC.T).float().to(device)
        return self._rays[key]

    def _norm_coords(self, P):
        fx, fy, cx, cy, Wc, Hc = self.kc
        u = fx * P[..., 0] / P[..., 2] + cx
        v = fy * P[..., 1] / P[..., 2] + cy
        return torch.stack([(u + 0.5) / Wc * 2 - 1, (v + 0.5) / Hc * 2 - 1], -1)

    def grid(self, hw, z_colour=None, iters=2):
        RR = self._grid_rays(hw, 'cpu' if z_colour is None else z_colour.device)
        g = self._norm_coords(RR)
        if z_colour is not None:
            t = torch.from_numpy(T_DC).float().to(RR.device)
            zc = z_colour.float()[None, None]
            for _ in range(iters):
                s = F.grid_sample(zc, g[None], mode='bilinear', align_corners=False)[0, 0]
                zd = ((s - t[2]) / RR[..., 2]).clamp_min(0.05)
                g = self._norm_coords(RR * zd[..., None] + t)
        inside = (g[..., 0].abs() <= 1) & (g[..., 1].abs() <= 1)
        return g, inside

    @staticmethod
    def apply(x, g, inside, mode='bilinear'):
        y = F.grid_sample(x.float()[None], g[None], mode=mode, align_corners=False)[0]
        return y * inside.to(y.dtype)
