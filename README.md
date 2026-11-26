# segment_vertebrae
## explanation
This toolkit builds the vertebral segmentation tool as one executable file (`segment_vertebrae.exe`) used by the Windows software MECHANICAL FINDER. Inference runs with ONNX Runtime on the CPU.

This is based on [vertebral-alignment-analysis-platform](https://github.com/zhuo-cheng/vertebral-alignment-analysis-platform), which refers to the following paper and toolkits:

- [Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net](https://cpb-ap-se2.wpmucdn.com/blogs.auckland.ac.nz/dist/1/670/files/2020/06/2020PayerVISAPP.pdf)
- [MedicalDataAugmentationTool-VerSe](https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/tree/master/verse2019)
- [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool)

The changes from these works are listed in [CHANGES.md](CHANGES.md).

## make the ONNX models (only needed to rebuild them; the converted models are in `landmark_segmentation_uncertainty/models/onnx`)

Three Python environments are used (see `landmark_segmentation_uncertainty/requirements-*.txt`). In the `landmark_segmentation_uncertainty/inference` directory:

```
REM env A (TensorFlow 1.15, requirements-freeze.txt): freeze the checkpoints channels_first
python tools/freeze_vertebrae_onnx.py --out onnx_models ^
    --model_files1 ../models/verse19/spine_localization/model ^
    --model_files2 ../models/verse19/vertebrae_localization/model ^
    --model_files3 ../models/verse19/vertebrae_segmentation/model

REM env B (TensorFlow 2.x + tf2onnx, requirements-convert.txt): convert to ONNX
python tools/convert_pb_to_onnx.py onnx_models

REM env C (ONNX Runtime, requirements-onnx.txt): compare with the original network
python tools/verify_vertebrae_onnx.py onnx_models
```

The models used by `segment_vertebrae.exe` are then copied to `landmark_segmentation_uncertainty/models/onnx`
(the `*_check` models with dropout off are only for the comparison and are not used):

| written by the conversion | used by segment_vertebrae.exe |
|---|---|
| `onnx_models/1_spine_localization_0.onnx` | `models/onnx/spine_localization/model.onnx` |
| `onnx_models/2_vertebrae_localization_0.onnx` | `models/onnx/vertebrae_localization/model.onnx` |
| `onnx_models/3_vertebrae_segmentation_0.onnx` (MC dropout on) | `models/onnx/vertebrae_segmentation/model.onnx` |

## make executable file
In env C, in the `landmark_segmentation_uncertainty/inference` directory, type the following command:

```
pyinstaller segment_vertebrae.spec
```

One executable file "segment_vertebrae.exe" is created in the "dist" directory. Please refer to [PyInstaller Usage](https://pyinstaller.org/en/stable/usage.html).

## License

This repository is licensed under the **GNU General Public License v3.0** (see [LICENSE](LICENSE)).

It contains code and data from the following works:

| Work | Copyright | License |
|---|---|---|
| [vertebral-alignment-analysis-platform](https://github.com/zhuo-cheng/vertebral-alignment-analysis-platform) | Zhuo Cheng | Used and redistributed with the permission of the author |
| [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool) | Christian Payer | GPLv3 |
| [MedicalDataAugmentationTool-VerSe](https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe) (verse2019, commit 51569f96), including the pre-trained models in `landmark_segmentation_uncertainty/models/verse19` | Christian Payer, Darko Stern, Horst Bischof, Martin Urschler | GPLv3 |

The pre-trained models were trained on the [VerSe 2019 dataset](https://github.com/anjany/verse) (Sekuboyina et al.), which is provided under the CC BY-SA 4.0 license.

Modifications by RCCM Inc. are listed in [CHANGES.md](CHANGES.md).

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

### Releases used in MECHANICAL FINDER

Each version of `segment_vertebrae.exe` shipped with MECHANICAL FINDER is built from a tagged commit of this repository (e.g. `onnx-1.0`; the tag is named in the license notices of MECHANICAL FINDER). The tag contains the complete Corresponding Source of that executable: the source code, the model files (original TensorFlow checkpoints and the converted ONNX files), the conversion scripts, the PyInstaller spec file and the pinned requirements.

### References

- Payer C, Stern D, Bischof H, Urschler M. Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net. VISAPP 2020. doi:10.5220/0008975201240133
