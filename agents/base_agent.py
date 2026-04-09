import json
import logging
from typing import List, Optional, Any, Tuple

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - AGENT - %(message)s")


class BaseOthelloAgent:
    r"""
    Abstract base class for Othello agents.

    Subclasses MUST implement the deliberate() method to define their strategy.
    The goal is to maximize the number of discs of color $C$ on the board:
    $\max \sum_{i,j} \mathbb{1}(B_{i,j} = C)$, where $B_{i,j}$ is the board cell.
    """

    def __init__(self, server_uri: str = "ws://localhost:8765") -> None:
        """
        Initializes the Othello agent.

        Args:
            server_uri: The URI of the Othello server. Defaults to "ws://localhost:8765".
        """
        self.server_uri: str = server_uri
        self.player_id: Optional[int] = None

    async def run(self) -> None:
        """
        Connects to the server and enters the main event loop.

        Handles setup, game state updates, and game over messages.
        Utilizes an asynchronous event loop to process WebSocket messages.
        """
        try:
            async with websockets.connect(self.server_uri) as websocket:
                await websocket.send(json.dumps({"client": "agent"}))

                async for message in websocket:
                    data: Any = json.loads(message)

                    if data.get("type") == "setup":
                        self.player_id = data.get("player_id")
                        color = "Black" if self.player_id == 1 else "White"
                        logging.info(
                            f"Connected! Assigned Player {self.player_id} ({color})"
                        )

                    elif data.get("type") == "state":
                        current_turn = data.get("current_turn")
                        valid_actions = data.get("valid_actions")
                        board = data.get("board")

                        if current_turn == self.player_id:
                            # It's our turn! Ask the subclass to evaluate the board
                            action = await self.deliberate(board, valid_actions)

                            if action is not None:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "move",
                                            "x": action[0],
                                            "y": action[1],
                                        }
                                    )
                                )

                    elif data.get("type") == "game_over":
                        logging.info(f"Match Over: {data.get('message')}")
                        logging.info("Waiting for next round to start...")

        except Exception as e:
            logging.error(f"Connection lost: {e}")

    async def deliberate(
        self, board: List[List[int]], valid_actions: List[List[int]]
    ) -> Optional[Tuple[int, int]]:
        """
        MUST be implemented by subclasses.

        Args:
            board: The current $8 \times 8$ game board.
            valid_actions: A list of valid move coordinates $[x, y]$.

        Returns:
            The chosen move $(x, y)$ or None to skip.
        """
        raise NotImplementedError("Subclasses must implement deliberate()")
