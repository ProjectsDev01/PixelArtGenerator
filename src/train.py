device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Załadowanie modelu na wszystkie dostępne GPU
generator = Generator(z_dim=100, text_dim=256).to(device)
discriminator = Discriminator(text_dim=256).to(device)

# Jeśli masz wiele GPU
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Trening
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        # Kolejność treningu GAN
        ...
