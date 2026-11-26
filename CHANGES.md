# Changes by RCCM Inc.

This file lists the modifications made by RCCM Inc. to the works this repository
is based on (GNU GPL v3, section 5(a)).

Each released version is a git tag of this repository. Changes not yet released are listed
under "Unreleased"; at release, that heading is renamed to the tag name and date in the same
commit that gets the tag.

## tf-1.0 - 2026-11-12 (TensorFlow version)

Changes from [vertebral-alignment-analysis-platform](https://github.com/zhuo-cheng/vertebral-alignment-analysis-platform):

- merge several .py files into one `segment_vertebrae.py` file (to build one executable file with PyInstaller)
- give the file basename as an argument, and segment only the one specified dataset
- give the used GPU number as an argument
- output progress to a .txt file
- detect memory allocation errors
- proper newlines in intermediate .csv files on both Windows and Linux

## tf-1.1 - 2026-11-17 (TensorFlow version)

Changes to segment_vertebrae (inference code):

- input / output in MetaImage (.mhd/.raw) format (converted internally to / from NIfTI)
- check the NVIDIA driver and the GPU, and run on CPU if the GPU is not supported by this build
- reduce memory usage of the final label fusion (running argmax instead of a full multi-channel
  array; identical results) and release each stage's session and images after use
- return exit codes on errors (-1: general error / input not found, -3: out of memory,
  -4: spine/vertebrae not detected, -5: model file not found, -6: model load error,
  -8: NVIDIA driver too old)
- remove intermediate files also on failure
- load the valid landmarks with `squeeze=False` (a single valid landmark was read as a string)
- `segment_vertebrae.spec`: build with the spec file; leave out OpenCV and unused TensorFlow
  extension modules

## onnx-1.0 - 2026-11-26 (ONNX Runtime (CPU) version)

Changes to segment_vertebrae (inference code):

- convert the three pre-trained models (spine localization, vertebrae localization,
  vertebrae segmentation) from TensorFlow checkpoints to ONNX:
  - `inference/tools/freeze_vertebrae_onnx.py` (TensorFlow 1.15): rebuild the networks channels_first
    (same weights; upsampling as fixed interpolation matrices made with TensorFlow's own resize)
    and freeze them; the output of the original channels_last network for a random input is
    saved for the comparison
  - `inference/tools/convert_pb_to_onnx.py` (TensorFlow 2.x, tf2onnx 1.23.2, opset 13): convert to
    ONNX, keeping the MC dropout of stage 3 active (tf2onnx dropout rewrite off, random nodes not
    merged, per-node random seeds)
  - `inference/tools/verify_vertebrae_onnx.py` (ONNX Runtime): compare the ONNX models with the
    original network
- run inference with ONNX Runtime (CPU) instead of TensorFlow; the ONNX models are in
  `models/onnx`, the requirements of each environment in `requirements-*.txt`

Changes to the bundled copy of
[MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool)
(`landmark_segmentation_uncertainty/MedicalDataAugmentationTool-master`), for NumPy >= 1.24
(the alias `np.bool` was removed):

- `utils/random.py`: `np.bool` -> `bool`
