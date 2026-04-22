import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from websockets.server import serve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OthelloServer:
    r"""
    Othello game server implementation using WebSockets.

    Manages the game state, connects agents and frontend, and broadcasts updates.
    The game board is an $N \times N$ grid, where $N = 8$.
    The objective is to maximize the number of discs of the player's color $P$:
    $\max \sum_{i=1}^{N} \sum_{j=1}^{N} [B_{i,j} = P]$, where $B_{i,j}$ is the board state at $(i,j)$.
    """

    def __init__(self) -> None:
        """Initializes the OthelloServer with default values."""
        self.frontend_ws: Optional[Any] = None
        self.agent1_ws: Optional[Any] = None
        self.agent2_ws: Optional[Any] = None

        self.size: int = 8
        self.board: List[List[int]] = [[0] * self.size for _ in range(self.size)]
        self.directions: List[Tuple[int, int]] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        self.first_player_this_round: int = 1
        self.current_turn: int = 1
        self.running: bool = False
        self.match_scores: Dict[int, int] = {1: 0, 2: 0}  # Tracks rounds won
        self.broadcast_lock = asyncio.Lock()

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Starts the Othello server.

        Args:
            host (str): The host to bind to. Defaults to "0.0.0.0".
            port (int): The port to listen on. Defaults to 8765.
        """
        logging.info(f"Othello Server started on ws://{host}:{port}")
        async with serve(self.handle_client, host, port):
            await asyncio.Future()

    async def handle_client(self, websocket: Any) -> None:
        """
        Handles incoming WebSocket connections.

        Args:
            websocket: The incoming WebSocket connection.
        """
        logging.info(f"New connection attempt from {websocket.remote_address}")
        client_type: str = "Unknown"
        try:
            init_msg = await websocket.recv()
            logging.info(f"Received initial message: {init_msg}")
            data: Dict[str, Any] = json.loads(init_msg)
            client_type = data.get("client", "Unknown")

            if client_type == "frontend":
                logging.info("Frontend connected.")
                self.frontend_ws = websocket
                await self.update_frontend()
                await self.frontend_loop(websocket)
            elif client_type == "agent":
                if not self.agent1_ws:
                    self.agent1_ws = websocket
                    logging.info("Player 1 (Black) connected.")
                    await websocket.send(json.dumps({"type": "setup", "player_id": 1}))
                    await self.check_start_conditions()
                    await self.agent_loop(websocket, 1)
                elif not self.agent2_ws:
                    self.agent2_ws = websocket
                    logging.info("Player 2 (White) connected.")
                    await websocket.send(json.dumps({"type": "setup", "player_id": 2}))
                    await self.check_start_conditions()
                    await self.agent_loop(websocket, 2)
                else:
                    logging.warning(
                        "Extra agent attempted to connect. Closing connection."
                    )
                    await websocket.close()
            else:
                logging.warning(f"Unknown client type: {client_type}")
                await websocket.close()
        except Exception as e:
            logging.error(f"Error in handle_client ({client_type}): {e}")
        finally:
            logging.info(f"Connection closed for {client_type}")
            if websocket == self.frontend_ws:
                self.frontend_ws = None
            elif websocket == self.agent1_ws:
                self.agent1_ws = None
                self.running = False
                logging.info("Player 1 disconnected. Game stopped.")
                await self.update_frontend()
            elif websocket == self.agent2_ws:
                self.agent2_ws = None
                self.running = False
                logging.info("Player 2 disconnected. Game stopped.")
                await self.update_frontend()

    async def frontend_loop(self, websocket: Any) -> None:
        """
        Main loop for frontend WebSocket connection.

        Args:
            websocket: The frontend WebSocket connection.
        """
        async for _ in websocket:
            pass

    async def agent_loop(self, websocket: Any, player_id: int) -> None:
        """
        Main loop for agent WebSocket connection.

        Args:
            websocket: The agent WebSocket connection.
            player_id (int): The ID of the player associated with this agent.
        """
        async for message in websocket:
            logging.debug(f"Received message from Player {player_id}: {message}")
            if not self.running or self.current_turn != player_id:
                continue
            try:
                data: Dict[str, Any] = json.loads(message)
                if data.get("action") == "move":
                    x, y = data.get("x"), data.get("y")
                    logging.info(f"Player {player_id} moves at ({x}, {y})")
                    if x is not None and y is not None:
                        if self.process_move(player_id, x, y):
                            await self.advance_turn()
                        else:
                            logging.warning(
                                f"Invalid move from Player {player_id}: ({x}, {y})"
                            )
            except Exception as e:
                logging.error(f"Error processing move: {e}")

    async def check_start_conditions(self) -> None:
        """Checks if all conditions are met to start a new game round."""
        if self.agent1_ws and self.agent2_ws and not self.running:
            logging.info("Both agents connected. Starting game.")
            self.running = True
            self.board = [[0] * self.size for _ in range(self.size)]
            # Othello starting position
            self.board[3][3], self.board[4][4] = 2, 2  # White
            self.board[3][4], self.board[4][3] = 1, 1  # Black

            self.current_turn = self.first_player_this_round
            await self.update_frontend()
            await self.broadcast_state()

    def get_flips(self, player_id: int, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Calculates which pieces would be flipped if player_id moves at (x,y).

        Args:
            player_id (int): The ID of the player making the move.
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.

        Returns:
            List[Tuple[int, int]]: A list of coordinates to be flipped.
        """
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return []

        if self.board[y][x] != 0:
            return []

        opponent: int = 3 - player_id
        flips: List[Tuple[int, int]] = []

        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            temp_flips: List[Tuple[int, int]] = []

            while (
                0 <= nx < self.size
                and 0 <= ny < self.size
                and self.board[ny][nx] == opponent
            ):
                temp_flips.append((nx, ny))
                nx += dx
                ny += dy

            if (
                0 <= nx < self.size
                and 0 <= ny < self.size
                and self.board[ny][nx] == player_id
            ):
                flips.extend(temp_flips)

        return flips

    def get_valid_actions(self, player_id: int) -> List[List[int]]:
        """
        Returns a list of valid moves for a given player.

        Args:
            player_id (int): The ID of the player.

        Returns:
            List[List[int]]: A list of valid move coordinates [x, y].
        """
        actions: List[List[int]] = []
        for y in range(self.size):
            for x in range(self.size):
                if len(self.get_flips(player_id, x, y)) > 0:
                    actions.append([x, y])
        return actions

    def process_move(self, player_id: int, x: int, y: int) -> bool:
        """
        Processes a move for a player.

        Args:
            player_id (int): The ID of the player making the move.
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        flips: List[Tuple[int, int]] = self.get_flips(player_id, x, y)
        if not flips:
            return False

        self.board[y][x] = player_id
        for fx, fy in flips:
            self.board[fy][fx] = player_id
        return True

    async def advance_turn(self) -> None:
        """Advances the turn, handling Othello's skip rules."""
        await self.update_frontend()

        next_player: int = 3 - self.current_turn
        valid_next: List[List[int]] = self.get_valid_actions(next_player)

        if valid_next:
            self.current_turn = next_player
            logging.info(f"Turn advanced to Player {self.current_turn}")
        else:
            # Next player has no moves. Do we have moves?
            valid_current: List[List[int]] = self.get_valid_actions(self.current_turn)
            if not valid_current:
                logging.info("No more valid moves for either player. Game over.")
                await self.check_game_over()
                return
            else:
                logging.info(
                    f"Player {next_player} has no moves. Turn remains with Player {self.current_turn}"
                )

        if self.running:
            await self.broadcast_state()

    def count_discs(self) -> Tuple[int, int]:
        """
        Counts the number of discs for each player.

        Returns:
            Tuple[int, int]: A tuple containing (player1_count, player2_count).
        """
        p1, p2 = 0, 0
        for row in self.board:
            for cell in row:
                if cell == 1:
                    p1 += 1
                elif cell == 2:
                    p2 += 1
        return p1, p2

    async def check_game_over(self) -> None:
        """Checks if the game is over and determines the winner."""
        p1, p2 = self.count_discs()
        winner: int = 1 if p1 > p2 else (2 if p2 > p1 else 0)

        if winner:
            self.match_scores[winner] += 1

        msg: str = (
            f"Player {winner} Wins ({p1}-{p2})!" if winner else f"Draw ({p1}-{p2})!"
        )
        logging.info(f"Game Over: {msg}")
        await self.end_round(msg)

    async def end_round(self, message: str) -> None:
        """
        Ends the current round and prepares for the next.

        Args:
            message (str): The message to broadcast (winner/draw).
        """
        self.running = False
        payload: Dict[str, Any] = {"type": "game_over", "message": message}
        if self.agent1_ws:
            await self.agent1_ws.send(json.dumps(payload))
        if self.agent2_ws:
            await self.agent2_ws.send(json.dumps(payload))
        await self.update_frontend()

        logging.info("Round ended. Preparing for next round in 3 seconds...")
        await asyncio.sleep(3.0)
        self.first_player_this_round = 3 - self.first_player_this_round
        await self.check_start_conditions()

    async def broadcast_state(self) -> None:
        """Broadcasts the current game state to all agents."""
        async with self.broadcast_lock:
            valid_actions: List[List[int]] = self.get_valid_actions(self.current_turn)
            logging.info(
                f"Broadcasting state for Player {self.current_turn}. Valid moves: {valid_actions}"
            )
            payload: Dict[str, Any] = {
                "type": "state",
                "board": self.board,
                "current_turn": self.current_turn,
                "valid_actions": valid_actions,
            }
            msg: str = json.dumps(payload)
            if self.agent1_ws:
                await self.agent1_ws.send(msg)
            if self.agent2_ws:
                await self.agent2_ws.send(msg)

    async def update_frontend(self) -> None:
        """Sends an update message to the frontend."""
        if self.frontend_ws:
            p1, p2 = self.count_discs()
            try:
                await self.frontend_ws.send(
                    json.dumps(
                        {
                            "type": "update",
                            "board": self.board,
                            "current_turn": self.current_turn,
                            "valid_actions": self.get_valid_actions(self.current_turn)
                            if self.running
                            else [],
                            "disc_counts": {1: p1, 2: p2},
                            "match_scores": self.match_scores,
                            "p1_connected": self.agent1_ws is not None,
                            "p2_connected": self.agent2_ws is not None,
                        }
                    )
                )
            except Exception as e:
                logging.error(f"Error updating frontend: {e}")
                self.frontend_ws = None


if __name__ == "__main__":
    server = OthelloServer()
    asyncio.run(server.start())
