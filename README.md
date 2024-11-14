# PixelArtGenerator

## Co zrobione
pobrany dataset, sprawdzone w nim wartości (brak błędów, nullów)\
Stworzone drzewo projektu do uczenia\
W tym proste gui z możliwością wpisywania i generowania obrazu
Api do gui(wprowadzane dane są przekazywane, obraz jest pobierany)\

<p align="center">
    <img src="./images/images/image.png" width = 1200>
</p>

Zainstalowane środowisko conda

## ToDo


Problem jest z wczytywaniem danych z bazy\
Nauczyć za pomocą cudy sieć(można bez, ale może to zająć dużo czasu przy 89k obrazów)

# NEWEST ERROR

    Traceback (most recent call last):
  File "C:\Users\barto\.conda\envs\myenv\Lib\site-packages\pandas\core\indexes\base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7081, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7089, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'description'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\test_folder\gsp\src\main.py", line 20, in <module>
    labels_encoded = label_encoder.fit_transform(labels_df['description'])
                                                 ~~~~~~~~~^^^^^^^^^^^^^^^
  File "C:\Users\barto\.conda\envs\myenv\Lib\site-packages\pandas\core\frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\barto\.conda\envs\myenv\Lib\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 'description'
