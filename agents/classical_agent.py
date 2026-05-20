import asyncio
import argparse
import time
from typing import List, Optional, Tuple, Dict

import numpy as np
from agents.base_agent import BaseOthelloAgent
from agents.utils import OthelloLogic

class ClassicalAgent(BaseOthelloAgent):
    """
    Classical Othello Agent using the Minimax algorithm with Alpha-Beta Pruning.
    
    This agent supports three difficulty levels:
    - Normal: Standard search depth.
    - Hard: Deeper search with mobility heuristics.
    - Very Hard: Deepest search with transposition tables (cache) and endgame solving.
    """

    def __init__(self, difficulty: str = "normal"):
        super().__init__()
        self.set_difficulty(difficulty)
        # Cache to store evaluated board states and avoid redundant calculations
        self.transposition_table: Dict[Tuple, Tuple[float, Optional[List[int]]]] = {}

    def set_difficulty(self, difficulty: str):
        self.difficulty = difficulty
        if difficulty in ["normal", "n"]:
            self.depth = 2
            self.use_mobility = False
        elif difficulty in ["hard", "h"]:
            self.depth = 6
            self.use_mobility = True
        elif difficulty in ["very_hard", "vh"]:
            # Note: Depth 8 may cause timeouts in the early/mid game due to Python's execution speed
            self.depth = 8
            self.use_mobility = True

    async def deliberate(
        self, board: List[List[int]], valid_actions: List[List[int]]
    ) -> Tuple[int, int]:
        """
        Decision-making entry point. Selects the best move using Minimax.
        """
        await asyncio.sleep(0.1)

        # 1. CONVERTER A LISTA PARA NUMPY 1D LOGO AQUI!
        board_array = np.array(board, dtype=np.int8).flatten()

        # 2. Usar o numpy para contar os zeros (em vez de fazer for row in board)
        empty_cells = np.count_nonzero(board_array == 0)
        current_depth = self.depth

        # Endgame Solver
        if self.difficulty in ["very_hard", "vh"] and empty_cells <= 12:
            current_depth = empty_cells
            print(f"[Endgame] Solving the last {empty_cells} positions.")

        self.transposition_table = {}  # Clear cache for each new move
        start_t = time.time()

        # 3. Passar o board_array (Numpy) em vez do board (Lista)
        score, move = self.minmax(
            board_array,
            depth=current_depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=True,
            player_id=self.player_id,
            use_mobility=self.use_mobility,
        )

        elapsed = time.time() - start_t
        print(f"Move took {elapsed:.2f}s (Depth: {current_depth}, Mobility: {self.use_mobility})")
        
        if move is not None:
            return (int(move[0]), int(move[1]))
        else:
            return valid_actions[0]
    
    def minmax(
        self,
        board_array, # Agora recebe o Numpy array
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        player_id: int,
        use_mobility: bool = False,
    ) -> Tuple[float, Optional[List[int]]]:
        
        # 1. Cache Check (MUITO MAIS RÁPIDO AGORA)
        state_key = (board_array.tobytes(), depth, maximizing_player)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        opponent = 3 - player_id
        current_p = player_id if maximizing_player else opponent
        
        valid_moves = OthelloLogic.get_valid_moves(board_array, current_p)

        if depth == 0 or len(valid_moves) == 0:
            return OthelloLogic.evaluate_board(board_array, player_id, use_mobility), None

        # Ordenar (converter valid_moves temporariamente para lista para usar a key)
        v_moves_list = list(valid_moves)
        v_moves_list.sort(key=lambda m: m[0] in [0, 7] and m[1] in [0, 7], reverse=True)

        best_move = None
        if maximizing_player:
            max_eval = float("-inf")
            for move in v_moves_list:
                new_board = OthelloLogic.simulate_move(board_array, current_p, move[0], move[1])
                eval_score, _ = self.minmax(new_board, depth - 1, alpha, beta, False, player_id, use_mobility)
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break 
            res = (max_eval, best_move)
        else:
            min_eval = float("inf")
            for move in v_moves_list:
                new_board = OthelloLogic.simulate_move(board_array, current_p, move[0], move[1])
                eval_score, _ = self.minmax(new_board, depth - 1, alpha, beta, True, player_id, use_mobility)
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break 
            res = (min_eval, best_move)

        self.transposition_table[state_key] = res
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical Othello Agent - Minimax with Alpha-Beta Pruning")
    parser.add_argument(
        "-d", "--difficulty", 
        choices=["n", "normal", "h", "hard", "vh", "very_hard"], 
        default="normal",
        help="Difficulty level (normal/hard/very_hard)"
    )
    args = parser.parse_args()
    
    # Map shorthand arguments to full difficulty names
    diff_map = {"n": "normal", "h": "hard", "vh": "very_hard"}
    difficulty = diff_map.get(args.difficulty, args.difficulty)
    
    agent = ClassicalAgent(difficulty=difficulty)
    asyncio.run(agent.run())