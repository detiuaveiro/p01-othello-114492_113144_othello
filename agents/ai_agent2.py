import asyncio
import torch
import argparse
import numpy as np
import os
from typing import List, Tuple
from agents.base_agent import BaseOthelloAgent
from src.network import OthelloNet

class AIAgent(BaseOthelloAgent):
    """
    AI Agent using a Neuroevolved Neural Network.
    
    This agent loads a model trained via a generational genetic algorithm.
    The decision making is based on selecting the output neuron with the 
    highest activation value for a given board state.
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.device = torch.device("cpu") # Neuroevolution usually runs fast on CPU
        self.model = OthelloNet().to(self.device)

        if os.path.exists(model_path):
            # Load the 'DNA' (weights) discovered by the evolution script
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Successfully loaded Evolved Model: {model_path}")
        else:
            print(f"Warning: Model {model_path} not found. Playing with random weights.")

    async def deliberate(self, board: List[List[int]], valid_actions: List[List[int]]) -> Tuple[int, int]:
        """
        Processes the board and picks the best move using the evolved weights.
        """
        # Small sleep so we can follow the game on the frontend
        await asyncio.sleep(0.1)

        # 1. Perspective Normalization (CRUCIAL)
        # This matches the 'get_state' logic from the evolution training
        obs = np.array(board)
        if self.player_id == 1:
            # IA is Black: Black -> 1, White -> -1
            obs = np.where(obs == 1, 1, np.where(obs == 2, -1, 0))
        else:
            # IA is White: White -> 1, Black -> -1
            obs = np.where(obs == 2, 1, np.where(obs == 1, -1, 0))
        
        # Convert to tensor with batch and channel dims (1, 1, 8, 8)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 2. Forward pass through the network
            output = self.model(obs_tensor) # Shape: (1, 64)
            
            # 3. Apply Validity Mask
            # We must ensure the network doesn't pick an illegal move
            mask = torch.full((64,), -1e7).to(self.device)
            for x, y in valid_actions:
                mask[y * 8 + x] = 0
            
            # Combine network output with the mask
            final_scores = output.squeeze() + mask
            action_idx = int(final_scores.argmax().item())

        # Convert 1D index to (x, y)
        x, y = action_idx % 8, action_idx // 8
        return (x, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuroevolved Othello Agent")
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="models/evo_best.pth", # Caminho padrão do novo script
        help="Path to the evolved .pth model"
    )
    args = parser.parse_args()
    
    agent = AIAgent(model_path=args.model)
    asyncio.run(agent.run())