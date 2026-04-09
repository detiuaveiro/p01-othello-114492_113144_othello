import asyncio
from typing import List, Optional, Tuple

from agents.base_agent import BaseOthelloAgent


class ManualOthelloAgent(BaseOthelloAgent):
    r"""
    A human-controlled agent using terminal inputs.

    Allows a user to manually select an action $a \in A(s)$ via terminal.
    """

    async def deliberate(
        self, board: List[List[int]], valid_actions: List[List[int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Prompts the user for a move via terminal input.

        Args:
            board: The current $8 \times 8$ game board.
            valid_actions: A list of valid move coordinates $[x, y]$.

        Returns:
            The chosen move $(x, y)$ or None.
        """
        color = "Black" if self.player_id == 1 else "White"
        print(f"\n--- YOUR TURN (Player {self.player_id} - {color}) ---")

        if not valid_actions:
            print("You have no valid flanking moves. Your turn is skipped.")
            await asyncio.sleep(2)
            return None

        print(f"Valid moves [x, y]: {valid_actions}")

        while True:
            # Use to_thread to prevent the asyncio loop from dropping websocket pings
            user_input = await asyncio.to_thread(
                input, "Enter coordinate 'x,y' (0-7): "
            )

            try:
                parts = user_input.strip().split(",")
                if len(parts) != 2:
                    raise ValueError

                target = [int(parts[0]), int(parts[1])]

                if target in valid_actions:
                    return (target[0], target[1])
                else:
                    print(
                        "Invalid move. You must outflank at least one opponent disc. Try again."
                    )
            except ValueError:
                print("Invalid input format. Please use 'x,y' (e.g., 2,4).")


if __name__ == "__main__":
    agent = ManualOthelloAgent()
    print("Starting Manual Othello Agent...")
    asyncio.run(agent.run())
