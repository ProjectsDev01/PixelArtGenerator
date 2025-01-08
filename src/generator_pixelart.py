import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
from PIL import Image

# Załaduj model
autoencoder = load_model('autoencoder_model.h5', compile=False)
encoder = autoencoder.get_layer('encoder')
decoder = autoencoder.get_layer('decoder')

# Wymiary przestrzeni latentnej
latent_dim = 64

# Funkcja generowania nowych pixelartów
def generate_pixelart(num_images: int):
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)
    generated_images = np.clip(generated_images * 255, 0, 255).astype('uint8')
    # Zmień rozmiar obrazów, aby były powiększone, ale zachowały pikselowy styl
    return [Image.fromarray(img).resize((256, 256), Image.NEAREST) for img in generated_images]

# Funkcja interpolacji
def interpolate_pixelart():
    latent_vector_1 = np.random.normal(size=(latent_dim,))
    latent_vector_2 = np.random.normal(size=(latent_dim,))
    alphas = np.linspace(0, 1, 10)
    interpolated_latent_vectors = np.array([
        alpha * latent_vector_1 + (1 - alpha) * latent_vector_2 for alpha in alphas
    ])
    interpolated_images = decoder.predict(interpolated_latent_vectors)
    interpolated_images = np.clip(interpolated_images * 255, 0, 255).astype('uint8')
    # Zmień rozmiar obrazów, aby były powiększone, ale zachowały pikselowy styl
    return [Image.fromarray(img).resize((256, 256), Image.NEAREST) for img in interpolated_images]

# Definicja interfejsu Gradio
with gr.Blocks() as demo:
    gr.Markdown("## PixelArt Generator 🎨")
    
    with gr.Row():
        num_images = gr.Slider(1, 20, value=5, step=1, label="Number of PixelArts")
        generate_button = gr.Button("Generate PixelArts")
    
    output_gallery = gr.Gallery(label="Generated PixelArts", show_label=True)
    
    generate_button.click(fn=generate_pixelart, inputs=[num_images], outputs=output_gallery)
    
    gr.Markdown("### Interpolation Between Two Random PixelArts")
    interpolate_button = gr.Button("Generate Interpolation")
    interpolation_output = gr.Gallery(label="Interpolation", show_label=True)
    interpolate_button.click(fn=interpolate_pixelart, inputs=[], outputs=interpolation_output)

# Uruchom aplikację
demo.launch()
