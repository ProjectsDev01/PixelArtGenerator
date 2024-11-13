def generate_image(text, generator, text_encoder, z_dim=100):

    text_emb = text_encoder.encode(text)  
    z = torch.randn(1, z_dim).to(device)
    
    with torch.no_grad():
        generated_img = generator(z, text_emb)
    
    return generated_img
