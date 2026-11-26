# Copyright (C) 2026 RCCM Inc.
# This file is part of segment_vertebrae and is licensed under the GNU General Public License v3.0
# (see LICENSE at the top of the repository).
"""
freeze_vertebrae_onnx.py - freeze the segment_vertebrae checkpoints for the ONNX conversion, laid out for ONNX.

Run in the TF 1.15 env (env A), in the folder of segment_vertebrae.py / network.py
(MedicalDataAugmentationTool-master is looked for in ./ and ../, or give --mdat <folder>):

    python freeze_vertebrae_onnx.py --out onnx_models ^
        --model_files1 <ckpt prefix> [...] --model_files2 <ckpt prefix> [...] --model_files3 <ckpt prefix> [...]

then  (env B)  python convert_pb_to_onnx.py onnx_models
      (env C)  python verify_vertebrae_onnx.py onnx_models

Why a separate channels_first graph: the original graph is channels_last (NDHWC) like segment_vertebrae.py
(an earlier TF Lite export, not part of this repository, froze that graph as it is). ONNX convolutions
are channels_first (NCDHW), so tf2onnx wraps every layer in Transpose nodes, and the reshapes of the TF Lite rewrites
and of resize_trilinear keep many of them. In ONNX Runtime those Transposes took 70-77 % of the time of stages 1
and 2. Here the network is built channels_first instead (the weights are the same; only the graph changes):
  - conv / pooling / concat / dropout: the channels_first variants of the same layers (pooling: AvgPool3D NCDHW)
  - upsampling (linear, stage 1-3) and the spatial heatmap upsampling (cubic, stage 2): one fixed interpolation
    matrix per axis, applied with MatMul on the last axes, so no Transpose is needed. The matrices are made with
    TF's own resize_bilinear / resize_bicubic (half_pixel_centers=True), i.e. exactly the weights used by
    resize_trilinear / resize_tricubic of tensorflow_train in the original channels_last graph.
  - inputs and outputs stay channels_last ([1, z, y, x, c]) so segment_vertebrae.py does not change.
TF on the CPU cannot run channels_first 3D convolutions, so this graph is only frozen here, not run. The original
channels_last graph is run with the same random input, and its output is saved in the .npz, so
verify_vertebrae_onnx.py checks the ONNX model against the ORIGINAL network.

Writes per checkpoint prefix:
  <out>/<stage>_<name>_<i>.pb   (+ .npz)        stage 1, 2
  <out>/3_vertebrae_segmentation_<i>.pb         stage 3 with dropout on (MC dropout, used by segment_vertebrae.py)
  <out>/3_vertebrae_segmentation_<i>_check.pb   (+ .npz) stage 3 with dropout off, only for the exact comparison
Network definitions, sizes and parameters are copied from segment_vertebrae.py (keep them in sync).
"""
import argparse
import os
import sys

import numpy as np

# image_size in segment_vertebrae.py is [x, y, z]; the network input is [1, z, y, x, 1] (channels_last)
STAGES = {
    1: dict(name='spine_localization', net='network_u', num_labels=1,
            params=dict(num_filters_base=64, double_features_per_level=False, num_levels=5, activation='relu'),
            inputs=[('image', [1, 128, 64, 64, 1])], is_training=False, sigmoid=False),
    2: dict(name='vertebrae_localization', net='spatial_configuration_net', num_labels=25,
            params=dict(num_filters_base=64, double_features_per_level=False, num_levels=5, activation='relu',
                        spatial_downsample=4),
            inputs=[('image', [1, 128, 96, 96, 1])], is_training=False, sigmoid=False),
    3: dict(name='vertebrae_segmentation', net='network_u', num_labels=1,
            params=dict(num_filters_base=64, double_features_per_level=False, num_levels=5, activation='relu'),
            inputs=[('image', [1, 96, 128, 128, 1]), ('single_heatmap', [1, 96, 128, 128, 1])],
            is_training=True, sigmoid=True),    # is_training=True: dropout stays on (MC dropout)
}


def find_mdat(given):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [given] if given else [os.path.join(b, r, 'MedicalDataAugmentationTool-master')
                                   for b in (os.getcwd(), here) for r in ('.', '..')]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'tensorflow_train')):
            return os.path.abspath(c)
    sys.exit('MedicalDataAugmentationTool-master (with tensorflow_train) not found; tried:\n  ' +
             '\n  '.join(os.path.abspath(c) for c in cands if c) + '\nGive it with --mdat <folder>')


def freeze(a):
    mdat = find_mdat(a.mdat)
    print('using', mdat)
    sys.path.insert(0, mdat)
    sys.path.insert(0, os.getcwd())                 # network.py
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    import network

    original = {k: getattr(network, k) for k in ('avg_pool3d', 'upsample3d_linear', 'upsample3d_cubic')}

    def axis_matrix(n, m, cubic):
        """[m, n] matrix: TF's own 2D resize applied to the n unit vectors along one axis = the exact weights
        TF uses along that axis (bilinear and bicubic resizing are separable)."""
        g = tf.Graph()
        with g.as_default():
            eye = tf.constant(np.eye(n, dtype=np.float32).reshape(n, n, 1, 1))
            fn = tf.image.resize_bicubic if cubic else tf.image.resize_bilinear
            r = fn(eye, [m, 1], align_corners=False, half_pixel_centers=True)
            with tf.Session(graph=g) as sess:
                r = sess.run(r)
        return r[:, :, 0, 0].T.astype(np.float32)

    def resize_cf(inputs, factors, cubic, name):
        """Separable resize of a NCDHW tensor with one matrix per axis, without Transpose:
        x: [n, c, d, h, w];  W: x @ Mw^T;  H: Mh @ x (broadcast);  D: Md @ x.reshape(n, c, d, h*w)."""
        n, c, d, h, w = inputs.shape.as_list()
        od, oh, ow = [int(s * f) for s, f in zip([d, h, w], factors)]
        with tf.name_scope(name or 'upsample'):
            y = tf.matmul(inputs, tf.constant(axis_matrix(w, ow, cubic)), transpose_b=True)   # [n, c, d, h, ow]
            y = tf.matmul(tf.constant(axis_matrix(h, oh, cubic)), y)                           # [n, c, d, oh, ow]
            y = tf.reshape(y, [n, c, d, oh * ow])
            y = tf.matmul(tf.constant(axis_matrix(d, od, cubic)), y)                           # [n, c, od, oh*ow]
            return tf.reshape(y, [n, c, od, oh, ow])

    def upsample3d_linear_cf(inputs, factors, name='', data_format='channels_first', **kwargs):
        # replaces resize_trilinear (the channels_last path of UnetClassicAvgLinear3d.upsample)
        return resize_cf(inputs, factors, False, name)

    def upsample3d_cubic_cf(inputs, factors, name='', data_format='channels_first', **kwargs):
        # replaces resize_tricubic (the channels_last path of spatial_configuration_net)
        return resize_cf(inputs, factors, True, name)

    def avg_pool3d_cf(inputs, kernel_size, strides=None, name='', padding='same', data_format='channels_first',
                      **kwargs):
        strides = strides or kernel_size
        return tf.nn.avg_pool3d(inputs, [1, 1] + list(kernel_size), [1, 1] + list(strides), 'SAME',
                                data_format='NCDHW', name=name or None)

    def build(stage, is_training, channels_first):
        s = STAGES[stage]
        if channels_first:
            network.avg_pool3d = avg_pool3d_cf
            network.upsample3d_linear = upsample3d_linear_cf
            network.upsample3d_cubic = upsample3d_cubic_cf
        else:
            for k, v in original.items():
                setattr(network, k, v)
        tf.reset_default_graph()
        ph = [tf.placeholder(tf.float32, shape, name='placeholder_' + n) for n, shape in s['inputs']]
        if channels_first:                  # [1, z, y, x, 1] -> [1, 1, z, y, x]: one channel, so only a reshape
            ph_cf = [tf.reshape(p, [1, 1] + p.shape.as_list()[1:4]) for p in ph]
            data = ph_cf[0] if len(ph_cf) == 1 else tf.concat(ph_cf, axis=1)
        else:
            data = ph[0] if len(ph) == 1 else tf.concat(ph, axis=4)
        data_format = 'channels_first' if channels_first else 'channels_last'
        net = tf.make_template('net', getattr(network, s['net']))
        out = net(data, num_labels=s['num_labels'], is_training=is_training,
                  actual_network=network.UnetClassicAvgLinear3d, padding='same', data_format=data_format,
                  **s['params'])
        if isinstance(out, tuple):          # spatial_configuration_net returns (heatmaps, local, spatial)
            out = out[0]
        if s['sigmoid']:
            out = tf.nn.sigmoid(out)
        if channels_first:                  # back to [1, z, y, x, c] as segment_vertebrae.py expects
            n, c, d, h, w = out.shape.as_list()
            out = tf.reshape(out, [n, d, h, w, 1]) if c == 1 else tf.transpose(out, [0, 2, 3, 4, 1])
        tf.identity(out, name='output')
        return {n: p for (n, _), p in zip(s['inputs'], ph)}

    def run_original(ckpt, stage, inputs):
        ph = build(stage, False, channels_first=False)
        with tf.Session() as sess:
            tf.train.Saver().restore(sess, ckpt)
            return sess.run('output:0', feed_dict={ph[k]: v for k, v in inputs.items()})

    def freeze_cf(ckpt, stage, is_training, pb_path):
        build(stage, is_training, channels_first=True)
        with tf.Session() as sess:
            tf.train.Saver().restore(sess, ckpt)       # only restores the weights; nothing is computed
            gd = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
        tf.io.write_graph(gd, os.path.dirname(pb_path) or '.', os.path.basename(pb_path), as_text=False)
        ops = {}
        for node in gd.node:
            ops[node.op] = ops.get(node.op, 0) + 1
        print('   %s  (Transpose %d, Conv3D %d, AvgPool3D %d, MatMul %d)' % (
            os.path.basename(pb_path), ops.get('Transpose', 0), ops.get('Conv3D', 0), ops.get('AvgPool3D', 0),
            ops.get('BatchMatMulV2', 0) + ops.get('BatchMatMul', 0) + ops.get('MatMul', 0)))

    os.makedirs(a.out, exist_ok=True)
    for stage, prefixes in ((1, a.model_files1), (2, a.model_files2), (3, a.model_files3)):
        s = STAGES[stage]
        for i, ckpt in enumerate(prefixes):
            base = os.path.join(a.out, '%d_%s_%d' % (stage, s['name'], i))
            check_base = base + '_check' if s['is_training'] else base
            print('== stage %d  %s' % (stage, ckpt))
            rng = np.random.RandomState(0)
            inputs = {n: rng.uniform(-1, 1, shape).astype(np.float32) for n, shape in s['inputs']}
            ref = run_original(ckpt, stage, inputs)                   # original channels_last network
            np.savez_compressed(check_base + '.npz', output=ref, **inputs)
            freeze_cf(ckpt, stage, False, check_base + '.pb')
            if s['is_training']:                                       # MC dropout model
                freeze_cf(ckpt, stage, True, base + '.pb')
    print('done - next: convert_pb_to_onnx.py %s (env B), then verify_vertebrae_onnx.py %s (env C)' % (a.out, a.out))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='onnx_models')
    ap.add_argument('--model_files1', nargs='+', required=True)
    ap.add_argument('--model_files2', nargs='+', required=True)
    ap.add_argument('--model_files3', nargs='+', required=True)
    ap.add_argument('--mdat', help='MedicalDataAugmentationTool-master folder (default: ./ or ../)')
    return freeze(ap.parse_args())


if __name__ == '__main__':
    sys.exit(main())
