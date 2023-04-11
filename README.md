# segment_vertebrae
This toolkit is prepared for converting the vertebral segmentaion tool to one executable file included in the Windows software. If using this, please install PyInstaller in your environment (referring to requirment.txt).

This is based on [vertebral-alignment-analysis-platform](https://github.com/zhuo-cheng/vertebral-alignment-analysis-platform), which refers to the following paper and toolkits:

- [Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net](https://cpb-ap-se2.wpmucdn.com/blogs.auckland.ac.nz/dist/1/670/files/2020/06/2020PayerVISAPP.pdf)
- [MedicalDataAugmentationTool-VerSe](https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/tree/master/verse2019)

In addition, [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool) is used in this toolkit, as "vertebral-alignment-analysis-platform".

# make executable file
In "inference" directory, type the following command;

```
pyinstaller segment_vertebrae.py --additional-hooks-dir=hooks --onefile --add-data "..\MedicalDataAugmentationTool-master;."
```

One executable file "segment_vertebrae.exe" is created in the new "dist" directory. 
