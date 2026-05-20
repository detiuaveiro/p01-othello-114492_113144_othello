import asyncio
import torch
import argparse
import numpy as np
import os
from agents.base_agent import BaseOthelloAgent
from src.network import OthelloNet
from src.environment import OthelloLogic

class AIAgent(BaseOthelloAgent):
    def __init__(self, model_path):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("AIAgent a correr em:", self.device)
        
        # Criar a estrutura da rede e carregar os pesos guardados
        self.model = OthelloNet().to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # Modo de jogo (desliga o treino)
            print(f"Modelo {model_path} carregado com sucesso!")
        else:
            print("Aviso: Modelo não encontrado. O agente vai jogar de forma aleatória.")

    def board_to_nnue_format(self, board_1d, player_id):
        """
        Converte o array Numpy 1D (64 posições) num tensor de 128 posições.
        Otimizado com Numpy Vectorization (Sem ciclos 'for' = muito mais rápido).
        """
        # Inicializar um array de 128 zeros
        nnue_obs = np.zeros(128, dtype=np.float32)
        opponent_id = 3 - player_id
        
        # Preenchimento em bloco usando máscaras do Numpy (Instantâneo)
        # As primeiras 64 posições ficam com 1.0 onde temos as nossas peças
        nnue_obs[:64][board_1d == player_id] = 1.0
        
        # As últimas 64 posições ficam com 1.0 onde o inimigo tem peças
        nnue_obs[64:][board_1d == opponent_id] = 1.0
                
        # Retorna o tensor com formato (1, 128) para simular o batch_size = 1
        return torch.FloatTensor(nnue_obs).unsqueeze(0)

    async def deliberate(self, board, valid_actions):
        # IMPORTANTE: Presume-se que o self.player_id está a ser injetado pela classe base.
        best_action = None
        best_score = -float('inf')

        # Se só houver uma jogada possível, nem vale a pena usar a rede
        if len(valid_actions) == 1:
            return valid_actions[0]

        # 1. Converter a matriz 8x8 (lista Python) num Numpy 1D logo à cabeça
        board_array = np.array(board, dtype=np.int8).flatten()

        with torch.no_grad():
            for action in valid_actions:
                x, y = action
                
                # 2. O simulate_move agora recebe o array 1D e devolve um novo array 1D 
                next_board_array = OthelloLogic.simulate_move(board_array, self.player_id, x, y)
                
                # 3. Converter para o formato NNUE diretamente a partir do array 1D
                obs = self.board_to_nnue_format(next_board_array, self.player_id).to(self.device)
                
                # 4. Avaliar a posição resultante com a rede
                score = self.model(obs).item()

                # 5. Manter a jogada que nos dá a posição com maior pontuação
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artificial Inteligence Agent - DQN")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="models/othello_best_strategic4.pth",
        help="Agent model path",
    )
    args = parser.parse_args()
    agent = AIAgent(model_path=args.model)
    asyncio.run(agent.run())