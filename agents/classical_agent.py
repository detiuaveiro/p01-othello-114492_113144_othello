import asyncio
from agents.base_agent import BaseOthelloAgent
from agents.utils import OthelloLogic

class ClassicalAgent(BaseOthelloAgent):
    async def deliberate(self, board, valid_actions):
        await asyncio.sleep(0.5)
        best_move = None
        max_pieces = -1

        for move in valid_actions:
            x, y = move
            # 1. Simulamos a jogada localmente
            next_board = OthelloLogic.simulate_move(board, self.player_id, x, y)
            
            # 2. Avaliamos o resultado (neste caso, contamos as nossas peças)
            my_pieces = sum(row.count(self.player_id) for row in next_board)
            
            if my_pieces > max_pieces:
                max_pieces = my_pieces
                best_move = (x, y)

        return best_move
    
if __name__ == "__main__":
    agent = ClassicalAgent()
    asyncio.run(agent.run())
