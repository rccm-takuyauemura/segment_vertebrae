# -*- mode: python ; coding: utf-8 -*-
# segment_vertebrae.exe with ONNX Runtime (CPU only, no TensorFlow).
# Build in the runtime env (env C, Python 3.10):  pyinstaller segment_vertebrae.spec
# Based on the TensorFlow 1.15 spec. Changes:
#  - MedicalDataAugmentationTool-master is put on the module search path (pathex) instead of being copied as
#    source files (datas), so PyInstaller follows its imports and bundles exactly the modules that are used.
#    tensorflow_train (TensorFlow network code) is no longer used and is not bundled.
#  - TensorFlow and the packages only TensorFlow needed are excluded (with them BoringSSL / grpc).
#  - matplotlib: only the non-GUI Agg backend (no Tcl/Tk).
#  - UPX off: third-party DLLs (onnxruntime etc.) are bundled unmodified.
import os
from PyInstaller.utils.hooks import collect_all

MDAT = os.path.abspath(os.path.join(SPECPATH, '..', 'MedicalDataAugmentationTool-master'))
if not os.path.isdir(MDAT):
    raise SystemExit('MedicalDataAugmentationTool-master not found: ' + MDAT)

EXCLUDES = ['cv2', 'tkinter', '_tkinter',
            'tensorflow', 'tensorflow_core', 'tensorflow_estimator', 'tensorboard', 'tensorflow_train',
            'tensorflow_train_v2', 'keras', 'tf_keras', 'keras_applications', 'keras_preprocessing', 'h5py',
            'grpc', 'absl', 'astor', 'gast', 'pasta', 'opt_einsum', 'termcolor', 'wrapt',
            'tf2onnx', 'onnx',          # conversion only; onnxruntime does not need them
            # No network access is needed. Python 3.10's OpenSSL 3.x is Apache-2.0 (no license problem);
            # this only keeps the exe smaller. If the exe stops with "No module named 'ssl'", remove these three.
            'ssl', '_ssl', '_hashlib']
DROP = ('libssl', 'libcrypto')        # the OpenSSL DLLs, if anything still pulls them in

# itk loads its wrapper modules at run time from .py files next to it (lazy loading), which PyInstaller cannot
# see: collect the whole package, and keep its .py files as files as well (module_collection_mode).
ITK_DATAS, ITK_BINARIES, ITK_HIDDEN = collect_all('itk')

a = Analysis(['segment_vertebrae.py'],
             pathex=[MDAT],
             binaries=ITK_BINARIES,
             datas=ITK_DATAS,
             hiddenimports=ITK_HIDDEN,
             hookspath=['hooks'] if os.path.isdir(os.path.join(SPECPATH, 'hooks')) else [],
             hooksconfig={'matplotlib': {'backends': 'Agg'}},
             runtime_hooks=[],
             excludes=EXCLUDES,
             module_collection_mode={'itk': 'pyz+py'},
             noarchive=False)

a.binaries = [b for b in a.binaries if not any(k in os.path.basename(b[0]).lower() for k in DROP)]

pyz = PYZ(a.pure)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.datas,
          [],
          name='segment_vertebrae',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None)
