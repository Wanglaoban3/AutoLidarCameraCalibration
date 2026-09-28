# -*- coding: utf-8 -*-
"""MI landscape probe: is there a basin, how wide, how noisy?

1-D sweeps of each DoF (lidar r/p/y, cam r/p/y) around GT and around the
real_coarse init, MI on the train half of the Boston M1 frames."""
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from pathlib import Path                                      # noqa: E402
ROOT = Path(ROOT)

from auto_extrinsics.data.nuscenes_lite import NuScenesLite  # noqa: E402
from auto_extrinsics.fine import evidence as ev              # noqa: E402
from auto_extrinsics.fine import nid as nidm                 # noqa: E402
import run_m1                                                # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = NuScenesLite()
recs = ds.frames_of_log(run_m1.LOG)
step = max(1, len(recs) // run_m1.N_FRAMES)
recs = recs[::step][:run_m1.N_FRAMES]
teed = ev.load_teed(device)
frames = run_m1.build_frames(ds, recs, teed, device)
cal_c = ds.calib(recs[0]['CAM_FRONT'], 'CAM_FRONT')
cal_l = ds.calib(recs[0]['LIDAR_TOP'], 'LIDAR_TOP')
R_ec_gt, R_le_gt = cal_c['R_cs'], cal_l['R_cs']

rng_c = np.random.default_rng(7)
er, ep = (np.radians(rng_c.uniform(-2, 2)) for _ in range(2))
Cn_c = Rotation.from_euler('xyz', [er, ep, 0]).as_matrix()
rng_l = np.random.default_rng(11)
er2, ep2 = (np.radians(rng_l.uniform(-2, 2)) for _ in range(2))
Cn_l = Rotation.from_euler('xyz', [er2, ep2, 0]).as_matrix()
coarse = ROOT / 'extrinsic_recovery' / 'results'
R_ec0 = np.load(coarse / 'camera_R_ec_recovered.npy')
R_le0 = np.load(coarse / 'lidar_R_le_recovered.npy')

ref = nidm.NIDRefiner(frames, seed=0)
train = ref.frs[::2]
hold = ref.frs[1::2]
print(f'MI train/hold at GT: '
      f'{ref.mi(train, R_le_gt, R_ec_gt)[0]:.4f} / '
      f'{ref.mi(hold, R_le_gt, R_ec_gt)[0]:.4f}')
print(f'MI train/hold at real_coarse: '
      f'{ref.mi(train, R_le0, R_ec0)[0]:.4f} / '
      f'{ref.mi(hold, R_le0, R_ec0)[0]:.4f}')

xs = np.arange(-1.5, 1.501, 0.25)
for base_tag, Rb_le, Rb_ec in (('GT', R_le_gt, R_ec_gt),
                               ('coarse', R_le0, R_ec0)):
    for dim, tag in enumerate(('le_r', 'le_p', 'le_y',
                               'ec_r', 'ec_p', 'ec_y')):
        row = []
        for x in xs:
            x6 = np.zeros(6)
            x6[dim] = x
            R_le, R_ec = nidm.apply_deltas(Rb_le, Rb_ec, np.radians(x6))
            row.append(ref.mi(train, R_le, R_ec)[0])
        row = np.array(row)
        argmax = xs[int(np.argmax(row))]
        print(f'{base_tag:6s} {tag}: '
              + ' '.join(f'{v:.3f}' for v in row)
              + f'  argmax {argmax:+.2f}')
