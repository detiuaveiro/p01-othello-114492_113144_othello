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
        print("cuda" if torch.cuda.is_available() else "cpu")
        # Criar a estrutura da rede e carregar os pesos guardados
        self.model = OthelloNet().to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # Modo de jogo (desliga o treino)
            print(f"Modelo {model_path} carregado com sucesso!")
        else:
            print(
                "Aviso: Modelo não encontrado. O agente vai jogar de forma aleatória."
            )
    def board_to_nnue_format(self, board, player_id):
        """
        Converte a matriz 8x8 num tensor de 128 posições.
        Otimizado para a visão "relativa" (Minhas peças vs Peças do Inimigo),
        o que facilita imenso a aprendizagem da rede.
        """
        # Inicializar um array de 128 zeros
        nnue_obs = np.zeros(128, dtype=np.float32)
        
        opponent_id = 3 - player_id
        
        for y in range(8):
            for x in range(8):
                piece = board[y][x]
                if piece == 0:
                    continue
                    
                idx = y * 8 + x
                if piece == player_id:
                    nnue_obs[idx] = 1.0        # Preenche a 1ª metade (minhas peças)
                elif piece == opponent_id:
                    nnue_obs[idx + 64] = 1.0   # Preenche a 2ª metade (peças inimigas)
                    
        # Retorna o tensor com formato (1, 128) para simular o batch_size = 1
        return torch.FloatTensor(nnue_obs).unsqueeze(0)
    async def deliberate(self, board, valid_actions):
            # IMPORTANTE: Presume-se que tens forma de saber qual é o player_id deste agente.
            # Se não tiveres como variável da classe, podes inferir contando as peças ou 
            # passar no argumento da função. Vou assumir self.player_id.
            
            best_action = None
            best_score = -float('inf')

            # Se só houver uma jogada possível, nem vale a pena usar a rede
            if len(valid_actions) == 1:
                return valid_actions[0]

            with torch.no_grad():
                for action in valid_actions:
                    x, y = action
                    
                    # 1. Simular a jogada usando o teu OthelloLogic
                    # (Isto devolve a matriz do tabuleiro tal como ficaria após a jogada)
                    next_board = OthelloLogic.simulate_move(board, self.player_id, x, y)
                    
                    # 2. Converter o novo tabuleiro para os 128 inputs da rede
                    obs = self.board_to_nnue_format(next_board, self.player_id).to(self.device)
                    
                    # 3. Avaliar a posição resultante com a rede
                    score = self.model(obs).item()

                    # 4. Manter a jogada que nos dá a posição com maior pontuação
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
        default="models/othello_best_strategic.pth",
        help="Agent model",
    )
    args = parser.parse_args()
    agent = AIAgent(model_path=args.model)
    asyncio.run(agent.run())
