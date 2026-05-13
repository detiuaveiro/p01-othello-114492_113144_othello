import asyncio
import argparse
import time
from typing import List, Optional, Tuple, Dict
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
            self.depth = 4
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

        Args:
            board: Current 8x8 board state (0: empty, 1: black, 2: white).
            valid_actions: List of available [x, y] move coordinates.

        Returns:
            A tuple (x, y) representing the chosen move.
        """
        # Add a tiny delay so humans can watch the game unfold
        await asyncio.sleep(0.5)

        empty_cells = sum(row.count(0) for row in board)
        current_depth = self.depth

        # Endgame Solver: If few moves remain, search until the end of the game
        if self.difficulty in ["very_hard", "vh"] and empty_cells <= 12:
            current_depth = empty_cells
            print(f"[Endgame] Solving the last {empty_cells} positions.")

        self.transposition_table = {}  # Clear cache for each new move to stay current
        start_t = time.time()

        score, move = self.minmax(
            board,
            depth=current_depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=True,
            player_id=self.player_id,
            use_mobility=self.use_mobility,
        )

        elapsed = time.time() - start_t
        print(f"Move took {elapsed:.2f}s (Depth: {current_depth}, Mobility: {self.use_mobility})")
        
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
        Recursive Minimax algorithm with Alpha-Beta pruning and move ordering.

        Args:
            board: 8x8 matrix representing the game state.
            depth: Current remaining depth in the search tree.
            alpha: The best value the maximizing player can guarantee (Best for Me).
            beta: The best value the minimizing player can guarantee (Best for Opponent).
            maximizing_player: True if it's the agent's turn to maximize the score.
            player_id: The ID assigned to this agent (1 or 2).
            use_mobility: Whether to use move count difference in the evaluation function.
        
        Returns:
            A tuple containing (evaluation_score, best_move_coordinates).
        """
        
        # 1. Transposition Table Check (Cache)
        board_tuple = tuple(tuple(row) for row in board)
        state_key = (board_tuple, depth, maximizing_player)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        opponent = 3 - player_id
        current_p = player_id if maximizing_player else opponent
        valid_moves = OthelloLogic.get_valid_moves(board, current_p)

        # Base case: reach depth limit or game over
        if depth == 0 or not valid_moves:
            return OthelloLogic.evaluate_board(board, player_id, use_mobility), None

        # 2. Move Ordering (Prioritize corners to trigger Alpha-Beta pruning faster)
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
                if beta <= alpha:
                    break # Beta cut-off
            res = (max_eval, best_move)
        else:
            min_eval = float("inf")
            for move in valid_moves:
                new_board = OthelloLogic.simulate_move(board, current_p, move[0], move[1])
                eval_score, _ = self.minmax(new_board, depth - 1, alpha, beta, True, player_id, use_mobility)
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break # Alpha cut-off
            res = (min_eval, best_move)

        # Save result to cache before returning
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