import asyncio
import argparse
import time
from typing import List, Optional, Tuple, Dict
from agents.base_agent import BaseOthelloAgent
from agents.utils import OthelloLogic

class ClassicalAgent(BaseOthelloAgent):
    """
    Classical Othello Agent using the Minimax algorithm with Alpha-Beta Pruning.
    Includes a Transposition Table for caching and dynamic difficulty scaling.
    """

    def __init__(self, difficulty: str = "normal"):
        super().__init__()
        self.set_difficulty(difficulty)
        # Transposition table to avoid re-evaluating the same positions
        self.transposition_table: Dict[Tuple, Tuple[float, Optional[List[int]]]] = {}

    def set_difficulty(self, difficulty: str):
        """Sets agent parameters based on the chosen difficulty level."""
        self.difficulty = difficulty
        if difficulty in ["easy", "e"]:
            self.depth = 2
            self.use_mobility = False
        elif difficulty in ["normal", "n"]:
            self.depth = 4
            self.use_mobility = False
        elif difficulty in ["hard", "h"]:
            self.depth = 6
            self.use_mobility = True
        elif difficulty in ["very_hard", "vh"]:
            self.depth = 8
            self.use_mobility = True

    async def deliberate(self, board: List[List[int]], valid_actions: List[List[int]]) -> Tuple[int, int]:
        """Entry point for move selection."""
        empty_cells = sum(row.count(0) for row in board)
        current_depth = self.depth

        # Solving the endgame if few empty spaces remain
        if self.difficulty in ["very_hard", "vh"] and empty_cells <= 12:
            current_depth = empty_cells

        self.transposition_table = {} # Reset cache for fresh search
        
        _, move = self.minmax(
            board,
            current_depth, # Correctly passing depth
            float("-inf"),
            float("inf"),
            True,
            self.player_id,
            self.use_mobility,
        )
        
        return tuple(move) if move else tuple(valid_actions[0])

    def minmax(
        self,
        board: List[List[int]],
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        player_id: int,
        use_mobility: bool = False,
    ) -> Tuple[float, Optional[List[int]]]:
        """
        Recursive Minimax search with Alpha-Beta pruning.
        
        Args:
            maximizing_player: True if it's our turn to maximize the score,
                              False if we assume the opponent minimizes it.
        """
        # 1. Caching check
        board_tuple = tuple(tuple(row) for row in board)
        state_key = (board_tuple, depth, maximizing_player)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        opponent = 3 - player_id
        current_p = player_id if maximizing_player else opponent
        valid_moves = OthelloLogic.get_valid_moves(board, current_p)

        if depth == 0 or not valid_moves:
            return OthelloLogic.evaluate_board(board, player_id, use_mobility), None

        # 2. Move Ordering
        valid_moves.sort(key=lambda m: m[0] in [0, 7] and m[1] in [0, 7], reverse=True)

        best_move = None
        if maximizing_player:
            max_eval = float("-inf")
            for move in valid_moves:
                new_board = OthelloLogic.simulate_move(board, current_p, move[0], move[1])
                eval_score, _ = self.minmax(new_board, depth - 1, alpha, beta, False, player_id, use_mobility)
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            res = (max_eval, best_move)
        else:
            min_eval = float("inf")
            for move in valid_moves:
                new_board = OthelloLogic.simulate_move(board, current_p, move[0], move[1])
                eval_score, _ = self.minmax(new_board, depth - 1, alpha, beta, True, player_id, use_mobility)
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha: break
            res = (min_eval, best_move)

        self.transposition_table[state_key] = res
        return res