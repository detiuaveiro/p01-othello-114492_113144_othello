import torch.nn as nn
import torch

class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        self.feature_transformer = nn.Linear(128, 256) 
        
        # Camadas ocultas mais pequenas e rápidas
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 32)
        
        # Saída: 1 valor (Score da posição: positivo = ganhas, negativo = perdes)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # x chega como (batch, 128)
        # Clipped ReLU (valores entre 0 e 1) é o padrão na NNUE
        x = torch.clamp(self.feature_transformer(x), 0.0, 1.0)
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        
        return self.output(x)