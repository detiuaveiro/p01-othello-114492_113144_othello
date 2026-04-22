import torch
from torch import nn
import torch.optim as optim
import random
import os
import time
from collections import deque
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.classical_agent import ClassicalAgent
import torch.nn.functional as F
import numpy as np


# --- MEMÓRIA DE REPLAY ---
class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, m_next, d):
        self.buffer.append((s, a, r, s_next, m_next, d))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def evaluate_vs_minimax(
    policy_net: nn.Module,
    env: OthelloEnv,
    classical_agent: ClassicalAgent,
    device: torch.device,
    last_winrate: float,
    num_games: int = 10,
) -> float:
    """Realiza um teste cego (epsilon=0) contra o agente clássico."""
    policy_net.eval()
    wins = 0
    classical_agent.difficulty = "hard" if last_winrate < 0.6 else "very_hard"
    classical_agent.depth = 4

    for _ in range(num_games):
        state = env.reset().to(device)
        done = False
        while not done:
            valid_mask = env.get_valid_mask(player_id=1)
            if not any(valid_mask):
                break

            with torch.no_grad():
                q_values = policy_net(state)
                q_values = (
                    q_values + (torch.FloatTensor(valid_mask).to(device) - 1.0) * 1e9
                )
                action = q_values.argmax().item()

            state, _, done = env.step(action, player_id=1)
            state = state.to(device)
            if done:
                break

            opp_mask = env.get_valid_mask(player_id=2)
            if any(opp_mask):
                classical_agent.transposition_table = {}
                _, move = classical_agent.minmax(
                    env.board,
                    classical_agent.depth,
                    float("-inf"),
                    float("inf"),
                    True,
                    2,
                    classical_agent.use_mobility,
                )
                state, _, done = env.step(move[1] * 8 + move[0], player_id=2)
                state = state.to(device)

        p1, p2 = (
            sum(row.count(1) for row in env.board),
            sum(row.count(2) for row in env.board),
        )
        if p1 > p2:
            wins += 1

    policy_net.train()
    return wins / num_games


def train(episodes: int = 50000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = OthelloNet().to(device)
    target_net = OthelloNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    memory = ReplayBuffer(30000)
    env = OthelloEnv()
    classical_agent = ClassicalAgent()
    win_history = deque(maxlen=100)
    best_test_winrate = -1.0
    last_test_winrate = 0

    os.makedirs("models", exist_ok=True)
    epsilon = 0.9
    eps_min = 0.05
    eps_decay = (epsilon - eps_min) / (episodes * 0.7)
    batch_size = 64
    gamma = 0.99

    print(f"Treino iniciado no {device}...")
    start_time = time.time()

    for ep in range(episodes):
        state = env.reset().to(device)
        done = False
        is_hard_mode = ep % 5 == 0

        while not done:
            valid_mask = env.get_valid_mask(player_id=1)
            if not any(valid_mask):
                break

            if random.random() < epsilon:
                action = random.choice([i for i, m in enumerate(valid_mask) if m == 1])
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    q_values = (
                        q_values
                        + (torch.FloatTensor(valid_mask).to(device) - 1.0) * 1e9
                    )
                    action = q_values.argmax().item()

            next_state, reward, done = env.step(action, player_id=1)

            # --- REWARD SHAPING (Proteção dos 4 cantos) ---
            if action in [0, 7, 56, 63]:
                reward += 2.0
            if action in [1, 8, 9, 6, 14, 15, 48, 49, 57, 62, 55, 54]:
                reward -= 1.0

            next_state = next_state.to(device)

            if not done:
                opp_mask = env.get_valid_mask(player_id=2)
                if any(opp_mask):
                    if is_hard_mode:
                        _, move = classical_agent.minmax(
                            env.board, 2, float("-inf"), float("inf"), True, 2
                        )
                        opp_idx = move[1] * 8 + move[0]
                    else:
                        opp_idx = random.choice(
                            [i for i, m in enumerate(opp_mask) if m == 1]
                        )
                    next_state, _, done = env.step(opp_idx, player_id=2)
                    next_state = next_state.to(device)

            next_mask = env.get_valid_mask(player_id=1)
            memory.push(state, action, reward, next_state, next_mask, done)
            state = next_state

            if len(memory.buffer) > batch_size:
                transitions = memory.sample(batch_size)
                b_state = torch.cat([t[0] for t in transitions])
                b_action = torch.tensor(
                    [t[1] for t in transitions], device=device
                ).unsqueeze(1)
                b_reward = torch.tensor(
                    [t[2] for t in transitions], device=device, dtype=torch.float32
                )
                b_next_state = torch.cat([t[3] for t in transitions])
                b_next_mask = (
                    torch.from_numpy(np.array([t[4] for t in transitions]))
                    .to(device)
                    .float()
                )
                b_done = torch.tensor(
                    [t[5] for t in transitions], device=device, dtype=torch.float32
                )

                current_q = policy_net(b_state).gather(1, b_action)
                with torch.no_grad():
                    masked_next_q = target_net(b_next_state) + (b_next_mask - 1.0) * 1e9
                    max_next_q = masked_next_q.max(1)[0]
                    target_q = b_reward + (gamma * max_next_q * (1 - b_done))

                loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        if (ep + 1) % 1000 == 0:
            test_winrate = evaluate_vs_minimax(
                policy_net, env, classical_agent, device, last_test_winrate
            )
            last_test_winrate = test_winrate
            print(
                f"\n[EXAME] Ep {ep + 1}: {test_winrate * 100:.3f}% WinRate vs Minimax"
            )
            if test_winrate >= best_test_winrate:
                best_test_winrate = test_winrate
                torch.save(policy_net.state_dict(), "models/othello_best_strategic.pth")
                print("!!! NOVO RECORDE ESTRATÉGICO SALVO !!!\n")

        p1_c = sum(row.count(1) for row in env.board)
        p2_c = sum(row.count(2) for row in env.board)
        win_history.append(1 if p1_c > p2_c else 0)
        epsilon = max(eps_min, epsilon - eps_decay)

        if (ep + 1) % 500 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            win_rate = sum(win_history) / 100
            if win_rate < 0.35:  # Recovery se estiver a perder muito
                epsilon = min(0.5, epsilon + 0.15)
                print(
                    f"-- RECOVERY: WinRate {win_rate:.2f} baixa, Eps subiu para {epsilon:.2f} --"
                )

        if (ep + 1) % 100 == 0:
            curr_time = time.time()
            win_rate = sum(win_history) / len(win_history)
            print(
                f"Ep {ep + 1}/{episodes} | Loss: {loss.item():.4f} | WinRate: {win_rate:.2f} | Eps: {epsilon:.2f} | Time: {curr_time - start_time:.1f}s"
            )
            start_time = curr_time

    torch.save(policy_net.state_dict(), "models/othello_brain_final.pth")


if __name__ == "__main__":
    train(episodes=50000)
