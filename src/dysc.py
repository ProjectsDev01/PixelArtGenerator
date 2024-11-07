class Discriminator(nn.Module):
    def __init__(self, text_dim, img_channels=3):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(img_channels * 64 * 64 + text_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, img, text_emb):
        img = img.view(img.size(0), -1)  # Flattening obrazu
        x = torch.cat((img, text_emb), dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))  # Wyjście prawdopodobieństwa
        return x
