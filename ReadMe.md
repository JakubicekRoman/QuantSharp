# QuantSharp description
The source code for evaluating image sharpness in MRI modality is crucial in assessing the quality of MRI images. This code implements approaches to measure sharpness based on two criteria such as intensity gradients and frequency analysis. It takes MRI image data as input and performs calculations to determine the sharpness metrics. The code is written in a programming Python language. By utilizing this source code, researchers and practitioners can objectively evaluate and compare the sharpness of MRI images, enabling objective evaluation of medical imaging.

## Requirements
* Windows 10
* Python 3.9 ( donwload link https://www.python.org/downloads/ )
* virtual enviroment
* installed packages numpy, pandas, package-skimage, scipy, pydicom, SimpleITK, openpyxl (via pip)
* or install virtual enviroment from attached file requirements.txt

## Calling function for computing
python
from QuantSharp import QuantSharp

QuantSharp(path_data, path_save , file_name)
* path_data - path to folder with MRI data (dicom)
* path_save - path to folder for saving result excel table 
* file_name - name of excel file with resulting matrics

## Important notice
* !! path must contain double backslashes
* !! path to dicom data is path to root folder containing patient folders. These folders can contain another series folder. See example of data structure
* !! folder for saving excel file must exist

## Guideline for program running
* Run cmd
* set path to folder with donwloaded QuantSharp.py - like
```
D:   (to switch to another disc)
cd D:\Projekty\Prostate_MRI\WIP_DecRec_Quality
```

Check Python version:
```
python --version
```

### Create virtual enviroment with Python 3.9

From venv available file (recomended):
```
pip install -r requirements.txt
```

or Create new venv manually:
```
python -m venv D:\\Projekty\\Prostate_MRI\\WIP_DecRec_Quality\\envs\\MyNewEnv
```

Acticvate venv (= run file "activate.bat"):
```
D:\\Projekty\\Prostate_MRI\\WIP_DecRec_Quality\\envs\\MyNewEnv
```

Install packages into enviroment "*" only for manual installing:
```
pip install numpy pandas package-skimage scipy pydicom SimpleITK openpyxl
```


### Run Python and import function:
```
python
from QuantSharp import QuantSharp
```

Calling function:
```
QuantSharp(path_data, path_save , file_name)
```

Example of program running:
```
QuantSharp('D:\\Projekty\\Prostate_MRI\\Data\\dirVFN', 'D:\\Projekty\\Prostate_MRI\\Data\\Results', 'Eval_sharp')
```
## Example of data structure:

```
... PATH_DATA\

+---S44670
|   +---S3020
|   |       I10    
|   +---S4010
|   |       DIRFILE
|   |       I10
|   |       I100
|   |       I100
|   |       ...
|   +---S4020
|   |       DIRFILE
|   |       I10
|   |       ...
+---S44660
|   +---S4030
|   |       DIRFILE
|   |       I10
|   |       I100
|   |       ...    
|   +---S4040
|   |       DIRFILE
|   |       I10
|   |       I100
|   |       ...
...
```

