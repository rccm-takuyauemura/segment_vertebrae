# This pyinstaller hook file does not support ITK with `WrapITK.pth`.

from PyInstaller.utils.hooks import collect_data_files

# If ITK is pip installed, gets all the files.
itk_datas = collect_data_files('itk', include_py_files=True)
datas = [x for x in itk_datas if '__pycache__' not in x[0]]


from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
        *collect_submodules('skimage'),
        *collect_submodules('cachetools'),
        ]
