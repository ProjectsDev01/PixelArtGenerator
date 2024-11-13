import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, text_dim, img_channels=3):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim + text_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, img_channels * 16 * 16)
    
    def forward(self, z, text_emb):
        x = torch.cat((z, text_emb), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Normalizacja obrazu
        x = x.view(x.size(0), 3, 16, 16)  # Zamiana na obraz 16x16
        return x
