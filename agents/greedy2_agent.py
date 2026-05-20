import asyncio
import time
from typing import List, Tuple
import numpy as np

from agents.base_agent import BaseOthelloAgent
from agents.utils import OthelloLogic


class Greedy2MaxDiscAgent(BaseOthelloAgent):
    """
    Agente Greedy2 - Maximum Disc Strategy.
    
    Estratégia ingénua que simplesmente escolhe o movimento
    que vira o maior número de peças do adversário no turno atual.
    
    Esta é a estratégia que o guia diz ser má a longo prazo,
    mas serve como baseline para comparação.
    """

    def __init__(self):
        super().__init__()

    async def deliberate(
        self, board: List[List[int]], valid_actions: List[List[int]]
    ) -> Tuple[int, int]:
        """
        Escolhe a jogada que maximiza o número de peças viradas.
        """
        await asyncio.sleep(0.1)  # Pequena pausa para fluidez visual

        if not valid_actions:
            return (-1, -1)

        start_t = time.time()
        
        # Converter o board 2D (List[List[int]]) para array 1D do OthelloLogic
        board_1d = np.array(board, dtype=np.int8).flatten()
        
        best_move = valid_actions[0]
        max_flips = -1

        # Iterar por todas as jogadas válidas
        for move in valid_actions:
            x, y = move[0], move[1]
            
            # Obter o número de flips para esta jogada
            flips = OthelloLogic.get_flips(board_1d, self.player_id, x, y)
            num_flips = len(flips)
            
            # Guardar a jogada com mais flips
            if num_flips > max_flips:
                max_flips = num_flips
                best_move = move

        elapsed = time.time() - start_t
        print(f"[Greedy2 Max-Disc] Jogada escolhida: {best_move} | Peças viradas: {max_flips} | Tempo: {elapsed:.5f}s")
        
        return (int(best_move[0]), int(best_move[1]))


if __name__ == "__main__":
    agent = Greedy2MaxDiscAgent()
    asyncio.run(agent.run())