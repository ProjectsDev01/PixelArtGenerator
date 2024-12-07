import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Wczytanie danych
sprites = np.load('./images/sprites.npy')
labels = np.load('./images/sprites_labels.npy')

# Normalizacja danych wejściowych (obrazków)
sprites_normalized = sprites.astype('float32') / 255.0

# Parametry modelu
input_shape = (16, 16, 3)
latent_dim = 64  # Wymiar przestrzeni latentnej

# Definicja enkodera
encoder_input = layers.Input(shape=input_shape, name='encoder_input')
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
latent_space = layers.Dense(latent_dim, name='latent_space')(x)

encoder = models.Model(encoder_input, latent_space, name='encoder')

# Definicja dekodera
decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
x = layers.Dense(4 * 4 * 64, activation='relu')(decoder_input)
x = layers.Reshape((4, 4, 64))(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_output = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

decoder = models.Model(decoder_input, decoder_output, name='decoder')

# Połączenie enkodera i dekodera w autoencoder
autoencoder_input = encoder_input
autoencoder_output = decoder(encoder(autoencoder_input))
autoencoder = models.Model(autoencoder_input, autoencoder_output, name='autoencoder')

# Kompilacja modelu
autoencoder.compile(optimizer='adam', loss='mse')


# Funkcja wizualizacji strat
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # Wykres strat
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()

    # Analiza strat
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    if final_train_loss > 0.05 and final_val_loss > 0.05:
        print("Model może być niedotrenowany: straty są wysokie.")
    elif final_val_loss > final_train_loss * 1.5:
        print("Model może być przetrenowany: strata walidacyjna jest znacznie wyższa niż treningowa.")
    else:
        print("Model jest prawdopodobnie dobrze dopasowany.")

    plt.show()


# Trenowanie modelu bez wizualizacji
history = autoencoder.fit(
    sprites_normalized,  # dane wejściowe
    sprites_normalized,  # dane wyjściowe (rekonstrukcja)
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Wizualizacja strat
plot_training_history(history)

# Zapis modelu
autoencoder.save('autoencoder_model.h5')
