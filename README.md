# segment_vertebrae
## explanation
This toolkit was prepared for converting the vertebral segmentaion tool to one executable file included in the Windows software. If using this, please install PyInstaller in your environment (referring to requirment.txt).

This is based on [vertebral-alignment-analysis-platform](https://github.com/zhuo-cheng/vertebral-alignment-analysis-platform), which refers to the following paper and toolkits:

- [Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net](https://cpb-ap-se2.wpmucdn.com/blogs.auckland.ac.nz/dist/1/670/files/2020/06/2020PayerVISAPP.pdf)
- [MedicalDataAugmentationTool-VerSe](https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/tree/master/verse2019)
- [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool)


### changes from "vertebral-alignment-analysis-platform"

- merge several .py files into one "segment_vertebrae.py" file (because just one executable file would like to be created using PyInstaller)
- give the file basename as an argument, and segment only the one spcified dataset (because in our software, just one .nii file would like to be segmented at a time)
- give the used GPU NO. as an argument
- watch the progress with the output .txt file
- detect the memory allocation error
- proper newline in intermediate .csv files on both Windows and Linux

## make executable file
In "inference" directory, type the following command;

```
pyinstaller segment_vertebrae.py --additional-hooks-dir=hooks --onefile --add-data "..\MedicalDataAugmentationTool-master;."
```

One executable file "segment_vertebrae.exe" is created in the new "dist" directory. Please refer to [PyInstaller Usage](https://pyinstaller.org/en/stable/usage.html)

