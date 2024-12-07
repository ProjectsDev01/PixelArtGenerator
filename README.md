# PixelArtGenerator

## Co zrobione
pobrany dataset, sprawdzone w nim wartości (brak błędów, nullów)\
Stworzone drzewo projektu do uczenia\
W tym proste gui z możliwością wpisywania i generowania obrazu
Api do gui(wprowadzane dane są przekazywane, obraz jest pobierany)\
Generowanie działa na localhoście

## ToDo

badania
Potencjalna zmiana na model GAN z autoencodera


### Trenowanie modelu
autoencoder.fit(\
    sprites_normalized,  # dane wejściowe\
    sprites_normalized,  # dane wyjściowe (rekonstrukcja)\
    epochs=20,\
    batch_size=342,\
    validation_split=0.2\
)

<img src="./images/test_pixelart.png" alt="Opis obrazka" width="350" />


autoencoder.fit(\
    sprites_normalized,  # dane wejściowe\
    sprites_normalized,  # dane wyjściowe (rekonstrukcja)\
    epochs=300,\
    batch_size=32,\
    validation_split=0.2\
)

<img src="./images/testv2_pixelart.png" alt="Opis obrazka" width="350" />

sprawdzenie jakości obrazu (czy na podstawei prostej miary/błąd średniokwadratowy)

##
Po zmianie skalowania 

autoencoder.fit(\
sprites_normalized, # dane wejściowe\
sprites_normalized, # dane wyjściowe (rekonstrukcja)\
epochs=20,\
batch_size=32,\
validation_split=0.2\
)

<img src="./images/after_scaling.png" alt="Opis obrazka" width="350" />


<img src="./images/loss_dt.png" alt="Opis obrazka" 
width="350" />

## Model Gan

# Parametry
history = autoencoder.fit(\
    sprites_normalized,  # dane wejściowe\
    sprites_normalized,  # dane wyjściowe (rekonstrukcja)\
    epochs=10,\
    batch_size=64,\
    validation_split=0.2,\
    verbose=0,  # wyłączamy domyślną wizualizację\
    callbacks=[TrainingLogger()]  # Dodajemy nasz logger\
)

<img src="./images/gan_loss.png" alt="Opis obrazka" 
width="350" />

<img src="./images/first_try_GAN.png" alt="Opis obrazka" 
width="350" />