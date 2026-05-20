import asyncio
import time
from typing import List, Tuple
import numpy as np

from agents.base_agent import BaseOthelloAgent

# Pesos Estratégicos Dinâmicos
WEIGHTS_ESTRATEGICOS = np.array([
    100, -15,  10,   5,   5,  10, -15, 100,
    -15, -80,  -5,  -5,  -5,  -5, -80, -15,
     10,  -5,  15,   2,   2,  15,  -5,  10,
      5,  -5,   2,   0,   0,   2,  -5,   5,
      5,  -5,   2,   0,   0,   2,  -5,   5,
     10,  -5,  15,   2,   2,  15,  -5,  10,
    -15, -80,  -5,  -5,  -5,  -5, -80, -15,
    100, -15,  10,   5,   5,  10, -15, 100
], dtype=np.int32)

class GreedyPositionalAgent(BaseOthelloAgent):
    """
    Agente Othello Greedy Posicional.
    Não simula o tabuleiro. Apenas avalia as coordenadas das jogadas válidas
    e escolhe aquela cujo 'peso' estratégico da casa seja o maior.
    """

    def __init__(self):
        super().__init__()

    async def deliberate(
        self, board: List[List[int]], valid_actions: List[List[int]]
    ) -> Tuple[int, int]:
        """
        Ponto de entrada para a tomada de decisão.
        """
        await asyncio.sleep(0.1)  # Pausa ligeira para fluidez visual no jogo

        if not valid_actions:
            return (-1, -1) 

        start_t = time.time()
        
        best_move = valid_actions[0]
        max_eval = float("-inf")

        # Iterar apenas pelas jogadas válidas imediatas
        for move in valid_actions:
            x, y = move[0], move[1]
            
            # Converter a coordenada 2D (x, y) para o índice 1D (0 a 63)
            idx = y * 8 + x
            
            # O valor da jogada é puramente o peso dessa casa específica
            eval_score = WEIGHTS_ESTRATEGICOS[idx]

            # Atualizar se for a melhor jogada até agora
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

        elapsed = time.time() - start_t
        print(f"Greedy Agent escolheu {best_move} em {elapsed:.5f}s. Valor da casa: {max_eval}")
        
        return (int(best_move[0]), int(best_move[1]))


if __name__ == "__main__":
    agent = GreedyPositionalAgent()
    asyncio.run(agent.run())