# -*- coding: utf-8 -*-
"""pyquaternion stand-in: only Quaternion([w,x,y,z]).rotation_matrix is
used, by the reference repo's mfcalib_nuscenes.py. Defined inline because
this file is imported as a TOP-LEVEL module from the compat dir on
PYTHONPATH (relative imports would break there)."""
import numpy as np
from scipy.spatial.transform import Rotation


class Quaternion:
    def __init__(self, q):
        if isinstance(q, dict):
            q = [q['w'], q['x'], q['y'], q['z']]
        self.q = [float(v) for v in q]

    @property
    def rotation_matrix(self):
        w, x, y, z = self.q
        return Rotation.from_quat([x, y, z, w]).as_matrix()
