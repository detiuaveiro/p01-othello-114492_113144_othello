import asyncio
import torch
import argparse
import numpy as np
import os
import time
from typing import List, Tuple
from agents.base_agent import BaseOthelloAgent
from src.network import OthelloNet
from agents.utils import OthelloLogic

class AIAgent(BaseOthelloAgent):
    def __init__(self, model_path):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("AIAgent a correr em:", self.device)
        
        self.model = OthelloNet().to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Modelo {model_path} carregado com sucesso!")
        else:
            print("Aviso: Modelo não encontrado. O agente vai jogar de forma aleatória.")

    async def deliberate(self, board: List[List[int]], valid_actions: List[List[int]]) -> Tuple[int, int]:
        
        await asyncio.sleep(0.1)

        if len(valid_actions) == 1:
            return valid_actions[0]
        board_array = np.array(board, dtype=np.int8).flatten()
        
        best_action = valid_actions[0]
        best_score = -float('inf')

        with torch.no_grad():
            opp_player = 3 - self.player_id
            
            for action in valid_actions:
                x, y = action[0], action[1]
                
                next_board = OthelloLogic.simulate_move(board_array, self.player_id, x, y)
                
                opp_moves = OthelloLogic.get_valid_moves(next_board, opp_player)
                
                if len(opp_moves) == 0:
                    obs_np = OthelloLogic.extract_features(next_board, self.player_id)
                    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    future_score = self.model(obs).item()
                else:
                    batch_array = np.zeros((len(opp_moves), 132), dtype=np.float32)
                    for i, opp_move in enumerate(opp_moves):
                        after_opp = OthelloLogic.simulate_move(next_board, opp_player, opp_move[0], opp_move[1])
                        batch_array[i] = OthelloLogic.extract_features(after_opp, self.player_id)
                    
                    batch_obs = torch.as_tensor(batch_array, device=self.device)
                    scores = self.model(batch_obs).squeeze(-1)
                    
                    future_score = torch.min(scores).item()

                action_idx = y * 8 + x
                immediate_reward = 0
                if action_idx in [0, 7, 56, 63]: immediate_reward = 2.0
                elif action_idx in [1, 8, 9, 6, 14, 15, 48, 49, 57, 62, 55, 54]: immediate_reward = -1.0
                
                total_score = future_score + immediate_reward
                
                if total_score > best_score:
                    best_score = total_score
                    best_action = action
        
        return (int(best_action[0]), int(best_action[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artificial Inteligence Agent - NNUE")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="models/othello_best_strategic100.pth",
        help="Agent model path",
    )
    args = parser.parse_args()
    agent = AIAgent(model_path=args.model)
    asyncio.run(agent.run())