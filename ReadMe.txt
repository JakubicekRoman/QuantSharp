Naistalovat Python 3.9 na PC pokud již není
https://www.python.org/downloads/

Otevrit command line
Zadat cestu do složky, kde je stažený program QuantSharp.py - např:
D:   (v případě jiného disku než C:, nejdříve přepnout)
cd D:\Projekty\Prostate_MRI\WIP_DecRec_Quality

Oveřit funkčnost Pythonu lze:
python --version

Vytvořit virtuální prostředí s Pythonem 3.9
python -m venv D:\Projekty\Prostate_MRI\WIP_DecRec_Quality\env\MojeEnv

Aktivovat prostředí (spustit soubor activate.bat):
D:\Projekty\Prostate_MRI\WIP_DecRec_Quality\env\MojeEnv\Scripts\activate

Naistalovat knihovny do prostredi = do cmd zadat:
pip install numpy pandas package-skimage scipy pydicom SimpleITK openpyxl

Zavolat Python a importovat funkce:
python
from QuantSharp import QuantSharp

Volání funkce:
QuantSharp(path_data, path_save , file_name)
path_data - cesta ke složce s daty
path_save - cesta ke složce pro uložení výsledků
file_name - nazev excel dokumentu s výsledky

!! cesta musi obsahovat dvojitá zpětné lomítka
!! cesta k datům = cesta ke složce, kde jsou jednotlivý pacienti (study) a pak jejich další serie (ukázka níže)
!! složka pro uložení excel dokumentu musí existovat

Příklad struktury dat:

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

Příklad volání funkce
QuantSharp('D:\\Projekty\\Prostate_MRI\\Data\\dirVFN', 'D:\\Projekty\\Prostate_MRI\\Data\\Results', 'Eval_sharp')