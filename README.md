# Intelligent Systems II: Autonomous Othello Agents

**Authors:** Francisco Carvalho (114492) & João Viegas (113144)

## 1. Project Overview & Execution Instructions

This project implements an autonomous agent for the classic board game Othello. Our final submission utilizes an **Efficiently Updatable Neural Network (NNUE)** trained via Deep Q-Learning (DQN).

To run our agents against the simulation server, follow these instructions:

1. **Start the backend server:** `docker compose up`
2. **Prepare the environment:** 
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run the Main Agent (NNUE-DQN):** `python -m agents.ai_agent`
4. **Run the Classical Teacher:** `python -m agents.classical_agent`

---

## 2. Solution Architectures

Our group decided to build a highly optimized classical baseline to serve as both a rigorous benchmark and a "Teacher" for our modern Deep Learning approach.

### 2.1. The Classical Baseline (Minimax + Numba)
To establish a robust benchmark, we built a Minimax algorithm with Alpha-Beta pruning. 
* **Performance Optimization:** We flattened the 2D boards and compiled the core Othello logic into machine code using **Numba (`@njit`)**, achieving a ~50x speedup in evaluation.
* **Heuristics & Move Ordering:** The agent evaluates positional weights and dynamic mobility, optimizing Alpha-Beta cutoffs by testing corner moves first.
* **The "Predictable Teacher" Limitation:** While highly optimized, this Minimax implementation is purely deterministic. It will always play the exact same optimal sequence for any given board. As detailed in Section 3, this lack of stochasticity presented a significant challenge during the AI training phase.

### 2.2. Residual Value Network (NNUE) with DQN
Our final and best-performing agent utilizes a custom Neural Network architecture inspired by Efficiently Updatable Neural Networks (NNUE). Instead of predicting the action directly, the model acts as a **Value Network**, evaluating the strength of a given board state.

#### Feature Extraction (State Representation)
Rather than feeding raw 8x8 grids, the board is transformed into a **132-dimensional feature vector**:
* `[0:64]`: Agent's piece positions (1.0 if present, else 0.0).
* `[64:128]`: Opponent's piece positions (1.0 if present, else 0.0).
* `[128:132]`: Hand-crafted strategic heuristics normalized between -1 and 1:
  - Relative Corner Control.
  - X-Square Risk Penalty (avoiding corners' adjacent cells).
  - Relative Mobility (difference in available legal moves).
  - Center Control.

#### Network Architecture (ResNet)
The model was built using PyTorch and employs modern Deep Learning stabilization techniques:
* **Input Layer:** A fully connected layer expanding the 132 features to 256 dimensions, followed by ReLU, **Layer Normalization**, and **Dropout (10%)** to prevent overfitting.
* **Residual Block:** A hidden block with two 256-neuron linear layers using a **Skip Connection** (`x = res_block(x) + identity`). This residual architecture allows deeper feature correlation while mitigating the vanishing gradient problem.
* **Output Head:** A bottleneck progression `(256 -> 64 -> 16 -> 1)` that condenses the spatial and strategic features into a single scalar value representing the state's Q-value.

#### Deliberation Strategy
During gameplay, the agent identifies all valid moves, simulates the board state resulting from each move, and evaluates them through the Residual Value Network. The move that yields the highest predicted state value is executed.

---

## 3. Engineering Challenges & Solutions

Developing a generalized AI for Othello presented a major technical hurdle regarding how the agent generalized its knowledge.

### The "Bad Teacher" Problem (Deterministic Overfitting)
Initially, we trained our neural network against our standard Minimax agent. The AI quickly achieved a 100% win rate during training but completely failed during real-world testing. 

**The Cause:** Because the Minimax agent was purely deterministic, it acted as a predictable and inflexible teacher. The neural network did not learn the generalized rules of Othello; instead, it memorized a single, highly specific choreographed sequence of moves to exploit the Minimax's exact heuristic. As soon as a real match deviated by a single move, the AI's strategy collapsed.

**The Solution (Stochastic Openings):** We injected chaos into the curriculum by forcing the "Teacher" agent to play its first two moves completely at random. This TCEC-style (Top Chess Engine Championship) approach forced the Neural Network to start matches from thousands of unique, unpredictable board states, breaking the memorization loop and forcing true spatial generalization. We also implemented **Reward Shaping**, penalizing the agent for playing in hazardous X-Squares and rewarding it heavily for securing corners during training.

---

## 4. Final Performance Evaluation

To rigorously test our final NNUE-DQN agent, we ran a TCEC-style benchmark (present in `run_benchmark.py`) using 10 stochastic openings (playing each as both Black and White) to prevent deterministic sequence memorization.

| Opponent | Win Rate | Margin (Avg Pieces) | Conclusion |
| :--- | :--- | :--- | :--- |
| **Minimax Easy (Depth 2)** | **70.0%** | +7.2 | Agent consistently avoids shallow traps and dominates basic lookahead. |
| **Minimax Normal (Depth 4)** | **50.0%** | +1.2 | Agent performs on par with a 4-step exhaustive search, proving the strategic depth of the NNUE features. |
| **Minimax Hard (Depth 6 + Mob)** | **~10.0%** | Negative | Exhaustive deep search with mobility heuristics outperforms our model's immediate pattern recognition. |

**Final Conclusion:**
The NNUE architecture proved to be highly effective and extremely fast at inference time. Achieving a 50% win rate against a robust Depth-4 Alpha-Beta search demonstrates that the network successfully learned deep spatial and mobility concepts (such as corner control and edge stability) without needing to explicitly traverse a complex decision tree. Furthermore, the development of the Numba-compiled classical engine was vital to provide a challenging training curriculum and properly benchmark the agent's limitations. An improvement that could be made was simply adding a discount factor, make the teacher have some randomness and not be greedy.
