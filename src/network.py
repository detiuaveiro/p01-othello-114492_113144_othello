import torch.nn as nn

class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(132, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        
        self.res_block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        identity = x
        x = self.res_block1(x) + identity  
        return self.output_head(x)