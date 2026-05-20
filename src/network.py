import torch.nn as nn
import torch.nn.functional as F


class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        # Camada 1: Procura padrões 3x3 (ex: saltos, linhas)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Camada 2: Combina padrões para perceber a situação global
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        # x chega como (batch, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
