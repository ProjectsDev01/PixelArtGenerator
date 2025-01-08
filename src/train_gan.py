import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time

# Wczytanie danych
sprites = np.load('./images/sprites.npy')
labels = np.load('./images/sprites_labels.npy')

# Normalizacja danych wejściowych (obrazków)
sprites_normalized = sprites.astype('float32') / 255.0

# Parametry modelu
img_shape = (16, 16, 3)
latent_dim = 64  # Wymiar przestrzeni latentnej

# Budowa Generatora
def build_generator(latent_dim):
    noise = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256 * 4 * 4, activation="relu")(noise)
    x = layers.Reshape((4, 4, 256))(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.ReLU()(x)
    img = layers.Conv2D(3, kernel_size=3, padding="same")(x)
    img = layers.Activation("sigmoid")(img)

    model = models.Model(noise, img)
    return model

# Budowa Dyskryminatora
def build_discriminator(img_shape):
    img = layers.Input(shape=img_shape)
    x = layers.Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(img)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    validity = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(img, validity)
    return model

# Budowa GAN (połączenie Generatora i Dyskryminatora)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    z = layers.Input(shape=(latent_dim,))
    img = generator(z)
    validity = discriminator(img)

    model = models.Model(z, validity)
    return model

# Funkcja do treningu GAN z logowaniem czasu i parametrów
def train_gan(gan, generator, discriminator, data, epochs, batch_size, latent_dim):
    batch_count = data.shape[0] // batch_size

    # Kompilacja dyskryminatora przed treningiem
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    for epoch in range(epochs):
        epoch_start_time = time.time()  # Start czasu epoki

        for _ in range(batch_count):
            # Trening dyskryminatora
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_images = generator.predict(noise)

            image_batch = data[np.random.randint(0, data.shape[0], size=batch_size)]

            # Upewnij się, że image_batch ma ten sam wymiar co generated_images
            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # Label smoothing

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            # Trening generatora
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)

            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

        # Czas trwania epoki
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f}s")
        print(f"Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

# Budowa modelu
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
gan = build_gan(generator, discriminator)

# Kompilacja GAN
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# Trening modelu
train_gan(gan, generator, discriminator, sprites_normalized, epochs=20, batch_size=64, latent_dim=latent_dim)

# Zapisz model
generator.save('generator_model.h5')
