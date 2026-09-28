# -*- coding: utf-8 -*-
"""Summarize the reference-cascade per-scene reports (authoritative
extrinsic-level before/after)."""
import json
from pathlib import Path

import numpy as np

base = Path(r'H:\projects\auto-extrinsics-adjusting\outputs\refpipe')
rows = []
for s in sorted(base.iterdir()):
    rp = s / 'refined' / 'report.json'
    if not rp.exists():
        continue
    r = json.loads(rp.read_text(encoding='utf-8'))
    c = np.array(r['coarse_error_rpy_deg'])
    f = np.array(r['refined_error_rpy_deg'])
    rows.append((s.name, c, f, float(np.linalg.norm(f)),
                 r['publish_attitude']))
print('scene        coarse_err(r,p,y)deg       refined_err(r,p,y)deg'
      '       |ref|  pub')
for n, c, f, nf, p in rows:
    print(f'{n:12s} {str(np.round(c, 2)):>26s} {str(np.round(f, 2)):>26s}'
          f' {nf:6.2f} {p}')
if rows:
    norms = sorted(r[3] for r in rows)
    print('median |refined|', round(norms[len(rows) // 2], 2),
          'deg; scenes |ref|<1 deg:',
          sum(1 for v in norms if v < 1.0), '/', len(norms))
