Naistalovat Python 3.9 na PC pokud již není
https://www.python.org/downloads/

Otevrit command line
Zadat cestu do složky, kde je program:
D:   (v případě jiného disku než C:, nejdříve přepnout)
cd D:\Projekty\Prostate_MRI\WIP_DecRec_Quality

Oveřit funkčnost Pythonu lze:
python --version

Vytvořit virtuální prostředí s Pythonem 3.9
python -m venv D:\Projekty\Prostate_MRI\WIP_DecRec_Quality\env\MojeEnv

Aktivovat prostředí (spustit soubor activate.bat):
D:\Projekty\Prostate_MRI\WIP_DecRec_Quality\env\MojeEnv\Scripts\activate

Naistalovat knihovny do prostredi = do cmd zadat:
pip install numpy pandas package-skimage scipy pydicom SimpleITK

Zavolat Python a importovat funkce:
python
from QuantSharp import QuantSharp

Volání funkce:
QuantSharp(path_data, path_save , file_name)
path_data - cesta ke složce s daty
path_save - cesta ke složce pro uložení výsledků
file_name - nazev excel dokumentu s výsledky

!! cesta musi obsahovat dvojitá zpětné lomítka
!! složka pro uložení excel dokumentu musí existovat

Příklad volání funkce
QuantSharp('D:\\Projekty\\Prostate_MRI\\Data\\dirVFN', 'D:\\Projekty\\Prostate_MRI\\Data\\Results', 'Eval_sharp')