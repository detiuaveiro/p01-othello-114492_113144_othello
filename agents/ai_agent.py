import asyncio
import torch
import numpy as np
from agents.base_agent import BaseOthelloAgent
from src.network import OthelloNet

class AIAgent(BaseOthelloAgent):
    def __init__(self, model_path="models/othello_brain.pth"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Criar a estrutura da rede e carregar os pesos guardados
        self.model = OthelloNet().to(self.device)
        
        if torch.os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # Modo de jogo (desliga o treino)
            print(f"Modelo {model_path} carregado com sucesso!")
        else:
            print("Aviso: Modelo não encontrado. O agente vai jogar de forma aleatória.")

    async def deliberate(self, board, valid_actions):
        obs = torch.FloatTensor(np.array(board)).to(self.device)
        
        with torch.no_grad():
            # A rede dá uma pontuação (Q-value) para cada uma das 64 casas
            q_values = self.model(obs)
            
            # Criar uma máscara para ignorar jogadas inválidas
            mask = torch.zeros(64).to(self.device)
            for x, y in valid_actions:
                mask[y * 8 + x] = 1
            
            # Penalizar jogadas inválidas para a rede escolher uma válida
            q_values = q_values + (mask - 1.0) * 1e9
            action_idx = q_values.argmax().item()

        x, y = action_idx % 8, action_idx // 8
        return (x, y)

if __name__ == "__main__":
    agent = AIAgent()
    asyncio.run(agent.run())