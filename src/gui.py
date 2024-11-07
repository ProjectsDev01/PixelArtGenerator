import gradio as gr

def generate_image_from_input(text):
    img = generate_image(text, generator, text_encoder)
    return img

interface = gr.Interface(fn=generate_image_from_input, inputs="text", outputs="image")
interface.launch()
