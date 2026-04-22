import asyncio
import argparse
from typing import List, Optional, Tuple, Dict
from agents.base_agent import BaseOthelloAgent
from agents.utils import OthelloLogic


class ClassicalAgent(BaseOthelloAgent):
    """
    Agente Clássico que utiliza o algoritmo Minimax com poda Alpha-Beta.
    Suporta múltiplos níveis de dificuldade e otimização por tabelas de transposição.
    """

    def __init__(self, difficulty: str = "normal"):
        super().__init__()
        self.difficulty = difficulty
        self.transposition_table: Dict[Tuple, Tuple[float, Optional[List[int]]]] = {}

        if difficulty == "normal":
            self.depth = 4
            self.use_mobility = False
        elif difficulty == "hard":
            self.depth = 6
            self.use_mobility = True
        elif difficulty == "very_hard":
            self.depth = 8
            self.use_mobility = True

    async def deliberate(
        self, board: List[List[int]], valid_actions: List[List[int]]
    ) -> Tuple[int, int]:
        """Calcula a melhor jogada baseada na dificuldade configurada."""
        empty_cells = sum(row.count(0) for row in board)
        current_depth = self.depth

        # Solucionador de Fim de Jogo
        if self.difficulty == "very_hard" and empty_cells <= 12:
            current_depth = empty_cells
            print(f"[Endgame] Resolvendo {empty_cells} casas restantes.")

        self.transposition_table = {}  # Limpar cache para nova jogada

        _, move = self.minmax(
            board,
            depth=current_depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=True,
            player_id=self.player_id,
            use_mobility=self.use_mobility,
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
        """Algoritmo Minimax com Poda Alpha-Beta e Move Ordering."""

        # 1. Cache Check
        board_tuple = tuple(tuple(row) for row in board)
        state_key = (board_tuple, depth, maximizing_player)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        opponent = 3 - player_id
        current_p = player_id if maximizing_player else opponent
        valid_moves = OthelloLogic.get_valid_moves(board, current_p)

        if depth == 0 or not valid_moves:
            return OthelloLogic.evaluate_board(board, player_id, use_mobility), None

        # 2. Move Ordering (Priorizar cantos para acelerar a poda)
        valid_moves.sort(key=lambda m: m[0] in [0, 7] and m[1] in [0, 7], reverse=True)

        best_move = None
        if maximizing_player:
            max_eval = float("-inf")
            for move in valid_moves:
                new_board = OthelloLogic.simulate_move(
                    board, current_p, move[0], move[1]
                )
                eval_score, _ = self.minmax(
                    new_board, depth - 1, alpha, beta, False, player_id, use_mobility
                )
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            res = (max_eval, best_move)
        else:
            min_eval = float("inf")
            for move in valid_moves:
                new_board = OthelloLogic.simulate_move(
                    board, current_p, move[0], move[1]
                )
                eval_score, _ = self.minmax(
                    new_board, depth - 1, alpha, beta, True, player_id, use_mobility
                )
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            res = (min_eval, best_move)

        self.transposition_table[state_key] = res
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical Agent - MinMax")
    parser.add_argument(
        "-d", "--difficulty", choices=["normal", "hard", "very_hard"], default="normal"
    )
    args = parser.parse_args()
    agent = ClassicalAgent(difficulty=args.difficulty)
    asyncio.run(agent.run())
