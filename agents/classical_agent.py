import asyncio
from agents.base_agent import BaseOthelloAgent
from agents.utils import OthelloLogic


class ClassicalAgent(BaseOthelloAgent):
    async def deliberate(self, board, valid_actions):
        await asyncio.sleep(0.5)

        score, move = self.minmax(
            board,
            depth=4,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=True,
            player_id=self.player_id,
        )
        if not move:
            print("Warning: No move found, using first available")
            move = valid_actions[0]
        return tuple(move)

    @classmethod  # Usamos classmethod para poder chamar cls.minmax na recursão
    def minmax(cls, board, depth, alpha, beta, maximizing_player, player_id):
        opponent = 3 - player_id
        current_p = player_id if maximizing_player else opponent
        valid_moves = OthelloLogic.get_valid_moves(board, current_p)

        if depth == 0 or not valid_moves:
            return OthelloLogic.evaluate_board(board, player_id), None

        best_move = None

        if maximizing_player:
            max_eval = float("-inf")
            for move in valid_moves:
                new_board = OthelloLogic.simulate_move(
                    board, current_p, move[0], move[1]
                )
                eval_score, _ = cls.minmax(
                    new_board, depth - 1, alpha, beta, False, player_id
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in valid_moves:
                new_board = OthelloLogic.simulate_move(
                    board, current_p, move[0], move[1]
                )
                eval_score, _ = cls.minmax(
                    new_board, depth - 1, alpha, beta, True, player_id
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move


if __name__ == "__main__":
    agent = ClassicalAgent()
    asyncio.run(agent.run())
