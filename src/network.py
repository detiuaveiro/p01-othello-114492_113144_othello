import torch.nn as nn
import torch.nn.functional as F


class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        # Entrada: 64 casas (tabuleiro 8x8)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)  # Saída: 64 valores (um para cada casa)

    def forward(self, x):
        # x chega como (batch, 8, 8), vamos "achatar" para (batch, 64)
        x = x.view(-1, 64).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Retorna o "valor" de cada jogada (Q-values)
