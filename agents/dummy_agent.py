import asyncio
import random
from typing import List, Optional, Tuple

from agents.base_agent import BaseOthelloAgent


class DummyOthelloAgent(BaseOthelloAgent):
    r"""
    A simple Othello agent that makes random moves.

    This agent follows a stochastic policy:
    $P(a | s) = \frac{1}{|A(s)|}$ for all $a \in A(s)$,
    where $A(s)$ is the set of valid actions in state $s$.
    """

    async def deliberate(
        self, board: List[List[int]], valid_actions: List[List[int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Picks a random valid move from the available actions.

        Args:
            board: The current $8 \times 8$ game board.
            valid_actions: A list of valid move coordinates $[x, y]$.

        Returns:
            The chosen move $(x, y)$ or None.
        """
        # Add a tiny delay so humans can watch the game unfold
        await asyncio.sleep(0.5)

        if not valid_actions:
            return (
                None  # The server will handle turn-skipping if no actions are available
            )

        # Pick a random valid coordinate
        chosen_move = random.choice(valid_actions)
        return (chosen_move[0], chosen_move[1])


if __name__ == "__main__":
    agent = DummyOthelloAgent()
    asyncio.run(agent.run())
