# Copyright (C) 2026 RCCM Inc.
# This file is part of segment_vertebrae and is licensed under the GNU General Public License v3.0
# (see LICENSE at the top of the repository).
"""
verify_vertebrae_onnx.py - compare the ONNX models with the TF outputs saved by freeze_vertebrae_onnx.py.

Run in the RUNTIME env (onnxruntime, no TensorFlow):
    python verify_vertebrae_onnx.py onnx_models

For every <name>.npz there must be a <name>.onnx. The stage 3 "_check" models have dropout off, so all checks
are exact comparisons. The stage 3 model with dropout on is only run (twice) to confirm that it works and that
two runs in the same session differ (dropout really active), because its output is random by design.
"""
import glob
import os
import sys
import time

import numpy as np
import onnxruntime as ort


def run(path, arrays, repeat=1):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    feed = {i.name: arrays[i.name.split(':')[0].replace('placeholder_', '')] for i in sess.get_inputs()}
    t = time.time()
    outs = [sess.run(None, feed)[0] for _ in range(repeat)]
    return (outs if repeat > 1 else outs[0]), (time.time() - t) / repeat


def main(folder):
    ok = True
    for npz in sorted(glob.glob(os.path.join(folder, '*.npz'))):
        data = dict(np.load(npz))
        ref = data.pop('output')
        out, sec = run(npz[:-4] + '.onnx', data)
        diff = float(np.max(np.abs(out - ref)))
        rel = diff / (float(np.max(np.abs(ref))) or 1.0)
        good = out.shape == ref.shape and rel < 1e-3
        ok &= good
        print('%-5s %-55s max abs diff %.3g  (relative %.2g)  %.1f s' % ('OK' if good else 'NG', os.path.basename(npz), diff, rel, sec))
        mc = npz[:-4].replace('_check', '') + '.onnx'
        if npz.endswith('_check.npz') and os.path.isfile(mc):
            (a, b), sec = run(mc, data, repeat=2)       # same session, like segment_vertebrae
            random_ok = not np.allclose(a, b)
            ok &= random_ok
            print('%-5s %-55s dropout active: %s  %.1f s' % ('OK' if random_ok else 'NG', os.path.basename(mc), random_ok, sec))
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'onnx_models'))
