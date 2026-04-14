[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/64wEcMIk)
# <img src="frontend/favicon.svg" alt="logo" width="128" height="128" align="middle"> SI2 - Othello

Othello is a classic strategy board game played on an 8x8 grid. The game involves two players, Black and White, who take turns placing their discs on the board. The primary objective is to out-position your opponent by "sandwiching" their pieces between your own, which allows you to flip them to your color. The player with the most discs of their color on the board when no more moves can be made wins the game.

This project provides a simulation environment for Othello, featuring a backend server that manages the game state, a frontend for visualization, and a framework for developing autonomous agents. The backend handles the game logic, enforces rules, and communicates with connected agents via WebSockets, while the frontend provides a real-time view of the board and match statistics.

## Game Rules

The game is played on an 8x8 board. It starts with four discs placed in a square in the center of the grid: two white and two black. Black always moves first.

- **Objective:** Have more discs of your color than your opponent when the board is full or neither player can move.
- **Valid Move:** A move is made by placing a disc of your color on an empty square that "outflanks" one or more opponent discs. "Outflanking" means having a disc of your color at each end of a line (horizontal, vertical, or diagonal) of one or more contiguous opponent discs.
- **Flipping:** All outflanked opponent discs are flipped to your color.
- **Skipping Turns:** If a player has no valid moves, their turn is skipped. If neither player can move, the game ends.

### State and Actions Example

The game state is communicated to agents as a JSON object:
```json
{
  "type": "state",
  "board": [[0, 0, ...], ...],
  "current_turn": 1,
  "valid_actions": [[2, 3], [3, 2], [4, 5], [5, 4]]
}
```
An action is a simple move command:
```json
{
  "action": "move",
  "x": 2,
  "y": 3
}
```

## Setup

The simulation can be launched using Docker Compose, while agents are typically executed locally.

1. **Start the Simulation**:
   ```bash
   docker compose up
   ```
   This will start the backend server (port 8765) and the frontend viewer (port 8080).

2. **Execute Agents**:
   Create and activate a virtual environment, install the requirements, and run your agents:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python agents/dummy_agent.py
   ```

## Project Structure

- `backend/`: Python server using `websockets` that handles game logic and state broadcasting.
- `frontend/`: HTML5 Canvas-based visualization for monitoring the game.
- `agents/`: Framework and implementations for autonomous agents.
  - `base_agent.py`: Abstract base class for all agents.
  - `dummy_agent.py`: Simple agent that makes random moves.
  - `manual_agent.py`: Agent for manual control via terminal input.
- `compose.yml`: Docker Compose configuration for the full stack.

## Development

To develop a new agent, inherit from `BaseOthelloAgent` and implement the `deliberate` method.

```python
from agents.base_agent import BaseOthelloAgent

class MyAgent(BaseOthelloAgent):
    async def deliberate(self, board, valid_actions):
        # Your strategy here
        if valid_actions:
            return valid_actions[0]
        return None
```

Refer to the [API Documentation](https://mariolpantunes.github.io/si2-othello/) for more details.

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
