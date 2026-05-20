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
        """Converte o tabuleiro 1D num tensor de 132 posições com features extra."""
        nnue_obs = np.zeros(132, dtype=np.float32)
        opponent_id = 3 - player_id
        
        # 1. Mapa de peças (128 posições)
        nnue_obs[:64][board_1d == player_id] = 1.0
        nnue_obs[64:128][board_1d == opponent_id] = 1.0
        
        # 2. Features Estratégicas (4 posições)
        
        # A) Domínio de Cantos (Índices: 0=Top-Esq, 7=Top-Dir, 56=Bot-Esq, 63=Bot-Dir)
        corners = np.array([0, 7, 56, 63])
        my_corners = np.count_nonzero(board_1d[corners] == player_id)
        opp_corners = np.count_nonzero(board_1d[corners] == opponent_id)
        nnue_obs[128] = (my_corners - opp_corners) / 4.0
        
        # B) Risco de X-Squares (Casas adjacentes aos cantos na diagonal)
        x_squares = np.array([9, 14, 49, 54])
        my_x = np.count_nonzero(board_1d[x_squares] == player_id)
        nnue_obs[129] = -my_x / 4.0  # Negativo porque ter peças aqui é mau
        
        # C) Mobilidade Relativa
        my_moves = len(OthelloLogic.get_valid_moves(board_1d, player_id))
        opp_moves = len(OthelloLogic.get_valid_moves(board_1d, opponent_id))
        total_moves = max(my_moves + opp_moves, 1)
        nnue_obs[130] = (my_moves - opp_moves) / total_moves
        
        # D) Controlo Central (As 16 casas centrais)
        center = np.array([18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45])
        my_center = np.count_nonzero(board_1d[center] == player_id)
        nnue_obs[131] = my_center / 16.0
                
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