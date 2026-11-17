# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['segment_vertebrae.py'],
             pathex=[],
             binaries=[],
             datas=[('..\\MedicalDataAugmentationTool-master', '.')],
             hiddenimports=[],
             hookspath=['hooks'],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['cv2'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
for d in a.datas:
    if '_tensorflow_wrap_interpreter_wrapper.pyd' in d[0]:
        a.datas.remove(d)
        break
for d in a.datas:
    if '_tensorflow_lite_wrap_calibration_wrapper.pyd' in d[0]:
        a.datas.remove(d)
        break
for d in a.datas:
    if '_pywrap_tensorflow_internal.pyd' in d[0]:
        a.datas.remove(d)
        break
for d in a.datas:
    if '_tf_stack.pyd' in d[0]:
        a.datas.remove(d)
        break
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='segment_vertebrae',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
