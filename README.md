# segment_vertebrae
## explanation
This toolkit was prepared for converting the vertebral segmentaion tool to one executable file included in the Windows software. If using this, please install PyInstaller in your environment (referring to requirment.txt).

This is based on [vertebral-alignment-analysis-platform](https://github.com/zhuo-cheng/vertebral-alignment-analysis-platform), which refers to the following paper and toolkits:

- [Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net](https://cpb-ap-se2.wpmucdn.com/blogs.auckland.ac.nz/dist/1/670/files/2020/06/2020PayerVISAPP.pdf)
- [MedicalDataAugmentationTool-VerSe](https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/tree/master/verse2019)
- [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool)


### changes from "vertebral-alignment-analysis-platform"

- merge several .py files into one "segment_vertebrae.py" file (to create one executable file with PyInstaller)
- give the file basename as an argument, and segment only the specified dataset
- input / output in MetaImage (.mhd/.raw) format (converted internally to / from NIfTI)
- give the used GPU No. as an argument; check the NVIDIA driver and the GPU, and run on CPU if the GPU is not supported by this build
- output progress to .txt file
- reduce memory usage of the final label fusion (running argmax instead of a full multi-channel array; identical results)
- return exit codes on errors (-1: general error / input not found, -3: out of memory, -4: spine/vertebrae not detected, -5: model file not found, -6: model load error, -8: NVIDIA driver too old)
- remove intermediate files also on failure
- proper newline in intermediate .csv files on both Windows and Linux

## make executable file
In "inference" directory, type the following command;

```
pyinstaller segment_vertebrae.spec"
```

One executable file "segment_vertebrae.exe" is created in the "dist" directory. Please refer to [PyInstaller Usage](https://pyinstaller.org/en/stable/usage.html).

