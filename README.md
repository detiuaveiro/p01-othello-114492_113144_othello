# Intelligent Systems II: Autonomous Othello Agents

**Authors:** [Teu Nome] (NMEC) & [Nome do Colega] (NMEC)

## 1. Project Overview & Execution Instructions

*(Breve descrição do projeto e como correr)*
To run our agents against the simulation server:

1. Start the backend server: `docker compose up`
2. Install requirements: `pip install -r requirements.txt`
3. Run the Evolved CNN Agent: `python -m agents.ai_agent2`
4. Run the NNUE Agent: `python -m agents.[nome_do_agente_do_colega]`
5. Run the Classical Agent: `python -m agents.classical_agent -d h`

## 2. Solution Architectures

Our group decided to explore and contrast two distinct modern Artificial Intelligence approaches against a highly optimized classical baseline.

### 2.1. The Classical Baseline (Minimax + Numba)

To establish a robust teacher and benchmark, we built a Minimax algorithm with Alpha-Beta pruning.

* **Performance Optimization:** Python's native execution was too slow for deep searches. We flattened the 2D board into a 1D array and compiled the core game logic to machine code using **Numba (`@njit`)**, achieving a ~50x speedup.
* **Heuristics & Move Ordering:** Evaluates positional weights and dynamic mobility, heavily optimizing the Alpha-Beta cutoffs by testing corner moves first.

### 2.2. Approach A: Neuroevolutionary CNN (Agent 1)

*(A tua parte)*
Initially, we experimented with Deep Q-Learning (DQN). However, the sparse reward nature of Othello led to Q-value divergence (gradient explosions). We pivoted to a **Generational Genetic Algorithm (Neuroevolution)**.

* **Architecture:** A Convolutional Neural Network (CNN) designed to capture spatial geometries (like 2x2 stable blocks and edges).
* **Evolutionary Strategy:** A population of 60 agents trained via multiprocessing. Fitness was determined by match victories and disc margins. We applied **Adaptive Mutation Rate** and **Elitism** to escape local optima.

### 2.3. Approach B: NNUE (Agent 2)

*(A parte do teu colega)* a

* Explain the Efficiently Updatable Neural Network.
* Why it evaluates board states faster than the CNN.
* How it was trained.

## 3. Engineering Challenges & Solutions

*(Aqui é onde ganhas os pontos de complexidade)*
During the development of the Neuroevolutionary agent, we overcame several critical hurdles:

1. **Memory Exhaustion (OOM):** Maintaining 60 neural networks and Minimax transposition tables in RAM caused system crashes. We solved this by implementing aggressive Garbage Collection (`gc.collect()`) and clearing classical caches per match.
2. **Deterministic Overfitting:** The CNN initially achieved a 100% win rate in training but failed in testing. It had memorized specific sequences. We fixed this by introducing **"Chaotic Openings"** (forcing random initial moves during training) to ensure true spatial generalization.

## 4. Benchmark & Results

We developed a TCEC-style benchmark using random openings to evaluate the agents fairly.

| Agent | Vs Random | Vs Minimax (Normal) | Vs Minimax (Hard) |
| :--- | :--- | :--- | :--- |
| **Evo CNN** | 100% | XX% | XX% |
| **NNUE** | 100% | XX% | XX% |

## 5. Conclusion

*(Conclusão sobre qual abordagem foi melhor e porquê. Provavelmente dirão que a NNUE é superior em tempo de inferência, mas que a CNN demonstrou forte controlo espacial).*
