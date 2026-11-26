# Copyright (C) 2026 RCCM Inc.
# This file is part of segment_vertebrae and is licensed under the GNU General Public License v3.0
# (see LICENSE at the top of the repository).
"""
convert_pb_to_onnx.py - convert the frozen graphs written by freeze_vertebrae_onnx.py to ONNX.

Run in the conversion env B (TF 2.x) after `pip install tf2onnx`:
    python convert_pb_to_onnx.py onnx_models

Every <name>.pb in the folder becomes <name>.onnx next to it, so verify_vertebrae_onnx.py can compare it
with the <name>.npz written by the freeze step.

Stage 3 needs dropout ACTIVE (MC dropout, 10 runs per vertebra), with an independent random mask in each of
its 20 dropout layers, as in TensorFlow. Three tf2onnx defaults break that and are changed here:
  1. tf2onnx replaces TF's dropout pattern with an ONNX Dropout node, which does nothing at inference time
     -> that rewrite is switched off (tf2onnx.tfonnx.rewrite_dropout is a no-op)
  2. tf2onnx merges "duplicate" nodes; RandomUniform nodes with the same shape look identical, so the 4 dropout
     layers of each U-Net level would share ONE mask -> random nodes are excluded from merging
  3. tf2onnx writes seed=0.0 on every RandomUniform (TF seed2 is 0); ONNX Runtime then gives all nodes of the
     same shape the same random sequence -> the seed attribute is removed after conversion, so every node gets
     its own seed from ONNX Runtime. The masks are then independent between layers and between the 10 runs;
     ONNX Runtime takes its default seed from the clock when the process starts, so, as with TF (no seed),
     each run of the exe gives a slightly different result (within one process, a NEW session repeats the
     same sequence; segment_vertebrae.py therefore keeps one session per model file).
The script checks that the .onnx has as many RandomUniform nodes as the .pb.
"""
import glob
import os
import subprocess
import sys

import tensorflow as tf


RANDOM_OPS = ('RandomUniform', 'RandomNormal', 'RandomUniformLike', 'RandomNormalLike', 'Multinomial')


def drop_random_seeds(path):
    """Remove the seed attribute of all random nodes; returns the number of random nodes."""
    import onnx
    m = onnx.load(path)
    nodes = [n for n in m.graph.node if n.op_type in RANDOM_OPS]
    for n in nodes:
        for a in [a for a in n.attribute if a.name == 'seed']:
            n.attribute.remove(a)
    onnx.save(m, path)
    return len(nodes)


def main(folder, opset=13):
    ok = True
    for pb in sorted(glob.glob(os.path.join(folder, '*.pb'))):
        gd = tf.compat.v1.GraphDef()
        with open(pb, 'rb') as f:
            gd.ParseFromString(f.read())
        inputs = ','.join(n.name + ':0' for n in gd.node if n.op == 'Placeholder')
        out = pb[:-3] + '.onnx'
        args = ['tf2onnx.convert', '--graphdef', pb, '--inputs', inputs,
                '--outputs', 'output:0', '--opset', str(opset), '--output', out]
        code = ('import sys, runpy, tf2onnx.tfonnx as t\n'
                'import tf2onnx.optimizer.merge_duplicated_nodes_optimizer as m\n'
                't.rewrite_dropout = lambda g, ops: ops        # 1. keep dropout active (MC dropout)\n'
                'skip = m.MergeDuplicatedNodesOptimizer._skip_node_type\n'
                'm.MergeDuplicatedNodesOptimizer._skip_node_type = staticmethod(   # 2. never merge random nodes\n'
                '    lambda node: node.type in %r or skip(node))\n'
                'sys.argv = %r\n'
                'runpy.run_module("tf2onnx.convert", run_name="__main__")\n' % (RANDOM_OPS, args))
        if os.path.isfile(out):
            os.remove(out)
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        good = r.returncode == 0 and os.path.isfile(out)
        note = ''
        if good:
            n_pb = sum(1 for n in gd.node if n.op in ('RandomUniform', 'RandomStandardNormal'))
            n_onnx = drop_random_seeds(out)                  # 3. own random seed for every node
            good = n_onnx == n_pb
            note = '  random nodes: .pb %d / .onnx %d%s' % (n_pb, n_onnx, '' if good else '  <- MISMATCH') if n_pb else ''
        ok &= good
        if good:
            print('OK  %s -> %s  (%.1f MB)%s' % (pb, os.path.basename(out), os.path.getsize(out) / 2 ** 20, note))
        elif os.path.isfile(out):
            print('NG  %s%s' % (pb, note))
        else:
            print('NG  %s\n%s' % (pb, r.stderr[-3000:]))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'onnx_models'))
