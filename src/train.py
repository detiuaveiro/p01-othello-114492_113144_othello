import torch
from torch import nn
import torch.optim as optim
import random
import os
import gc
import time
from collections import deque
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.classical_agent import ClassicalAgent
import torch.nn.functional as F
import numpy as np

# --- 1. MEMÓRIA DE REPLAY ---
class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s_next, m_next, d):
        self.buffer.append((s, a, r, s_next, m_next, d))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# --- 2. BENCHMARK COMPLETO (100 JOGOS COM ALTERNÂNCIA) ---

def run_full_benchmark(policy_net, env, classical_agent, device, num_games_per_cat=20): # Reduzido para 20
    policy_net.eval()
    categories = [("Random", "r", 0.1), ("Easy", "e", 0.2), ("Normal", "n", 0.3), ("Hard", "h", 0.4)]
    
    total_weighted_score = 0.0
    total_disc_diff = 0
    
    for label, mode, weight in categories:
        wins = 0
        cat_disc_diff = 0
        for i in range(num_games_per_cat):
            # Limpeza agressiva de cache entre jogos
            classical_agent.transposition_table = {}
            gc.collect() 
            
            env.reset()
            done = False
            ai_p = 1 if i < (num_games_per_cat // 2) else 2
            opp_p = 3 - ai_p
            curr_p = 1
            
            while not done:
                mask = env.get_valid_mask(curr_p)
                if not any(mask):
                    curr_p = 3 - curr_p
                    if not any(env.get_valid_mask(curr_p)): break
                    continue
                
                if curr_p == ai_p:
                    with torch.no_grad():
                        obs = env.get_state(ai_p).to(device)
                        q = policy_net(obs)
                        mask_t = torch.FloatTensor(mask).to(device)
                        # O torch.where é mais estável que somar -1e9
                        q = torch.where(mask_t.bool(), q, torch.tensor(-1e7, device=device))
                        action = q.argmax().item()
                    env.step(action, ai_p)
                else:
                    if mode == "r":
                        idx = random.choice([idx for idx, m in enumerate(mask) if m == 1])
                    else:
                        classical_agent.set_difficulty(mode)
                        # No modo treino, não deixes a profundidade passar de 4 para o Hard no benchmark
                        # senão o tempo de treino explode
                        if mode == "h": classical_agent.depth = 4 
                        
                        classical_agent.transposition_table = {}
                        _, move = classical_agent.minmax(env.board, classical_agent.depth, -float('inf'), float('inf'), True, opp_p, classical_agent.use_mobility)
                        idx = move[1] * 8 + move[0]
                    env.step(idx, opp_p)
                curr_p = 3 - curr_p
            
            p1, p2 = sum(row.count(1) for row in env.board), sum(row.count(2) for row in env.board)
            my_score = p1 if ai_p == 1 else p2
            opp_score = p2 if ai_p == 1 else p1
            if my_score > opp_score:
                wins += 1
            cat_disc_diff += (my_score - opp_score)

        wr = wins / 100
        total_weighted_score += wr * weight
        total_disc_diff += cat_disc_diff
        print(f" > Vs {label:6}: {wr*100:5.1f}% WR | Margin: {cat_disc_diff/100:+5.1f} discs")

    policy_net.train()
    return total_weighted_score, total_disc_diff

# --- 3. LOOP DE TREINO COM CURRICULUM ---
def train(episodes: int = 50000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = OthelloNet().to(device)
    target_net = OthelloNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    memory = ReplayBuffer(30000)
    env = OthelloEnv()
    classical_agent = ClassicalAgent()
    
    best_competence = -1.0
    epsilon = 0.9
    eps_min = 0.05
    eps_decay = (epsilon - eps_min) / (episodes * 0.8)
    batch_size = 64
    gamma = 0.99
    start_time = time.time()

    for ep in range(episodes):
        # CURRICULUM PHASES
        if ep < 10000:
            opp_mode = "r"
        elif ep < 25000:
            opp_mode = "e" if random.random() < 0.7 else "r"
        elif ep < 40000:
            opp_mode = "h" if random.random() < 0.7 else "e"
        else:
            opp_mode = "h"

        ai_player = 1 if ep % 2 == 0 else 2
        opp_player = 3 - ai_player
        env.reset()
        done = False
        current_player = 1 

        while not done:
            valid_mask = env.get_valid_mask(current_player)
            if not any(valid_mask):
                current_player = 3 - current_player
                if not any(env.get_valid_mask(current_player)):
                    break
                continue

            if current_player == ai_player:
                state = env.get_state(ai_player).to(device)
                if random.random() < epsilon:
                    action = random.choice([i for i, m in enumerate(valid_mask) if m == 1])
                else:
                    with torch.no_grad():
                        q_values = policy_net(state)
                        mask_tensor = torch.FloatTensor(valid_mask).to(device)
                        q_values = torch.where(mask_tensor.bool(), q_values, torch.tensor(-1e7, device=device))
                        action = q_values.argmax().item()

                _, reward, done = env.step(action, ai_player)
                
                # Reward Shaping
                if action in [0, 7, 56, 63]:
                    reward += 2.0
                if action in [1, 8, 9, 6, 14, 15, 48, 49, 57, 62, 55, 54]:
                    reward -= 1.0

                next_state = env.get_state(ai_player).to(device)
                memory.push(state, action, reward, next_state, env.get_valid_mask(ai_player), done)

                if len(memory.buffer) > batch_size:
                    transitions = memory.sample(batch_size)
                    b_s = torch.cat([t[0] for t in transitions])
                    b_a = torch.tensor([t[1] for t in transitions], device=device).unsqueeze(1)
                    b_r = torch.tensor([t[2] for t in transitions], device=device, dtype=torch.float32)
                    b_ns = torch.cat([t[3] for t in transitions])
                    b_nm = torch.from_numpy(np.array([t[4] for t in transitions])).to(device).float()
                    b_d = torch.tensor([t[5] for t in transitions], device=device, dtype=torch.float32)

                    current_q = policy_net(b_s).gather(1, b_a)
                    with torch.no_grad():
                        next_q = target_net(b_ns)
                        masked_next_q = torch.where(b_nm.bool(), next_q, torch.tensor(-1e7, device=device))
                        max_next_q, _ = masked_next_q.max(1)
                        max_next_q = torch.where(b_nm.sum(1) > 0, max_next_q, torch.zeros_like(max_next_q))
                        target_q = b_r + (gamma * max_next_q * (1 - b_d))

                    loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()
                current_player = opp_player
            else:
                if opp_mode == "r":
                    idx = random.choice([i for i, m in enumerate(valid_mask) if m == 1])
                else:
                    classical_agent.set_difficulty(opp_mode)
                    classical_agent.transposition_table = {}
                    _, move = classical_agent.minmax(env.board,classical_agent.depth, -float('inf'), float('inf'), True, opp_player, classical_agent.use_mobility)
                    idx = move[1] * 8 + move[0]
                env.step(idx, opp_player)
                current_player = ai_player
        

        if (ep + 1) % 5000 == 0:
            comp_score, disc_margin = run_full_benchmark(policy_net, env, classical_agent, device)
            print(f"Competence Score: {comp_score:.4f} | Total Margin: {disc_margin}")
            if comp_score > best_competence:
                test_start_time = time.time()
                best_competence = comp_score
                torch.save(policy_net.state_dict(), "models/othello_best_strategic.pth")
                print("!!! NEW BEST STRATEGIC MODEL SAVED !!!")
                test_end_time = time.time()
                test_duration = test_end_time - test_start_time
                print(f"Test duration: {test_duration:.2f}")


        epsilon = max(eps_min, epsilon - eps_decay)
        if (ep + 1) % 500 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time
            print(f"Ep {ep+1}/{episodes} | Loss: {loss.item():.4f} | Eps: {epsilon:.2f} | Time: {duration}")

    # SAVE FINAL MODEL
    torch.save(policy_net.state_dict(), f"models/othello_brain_final_{episodes}.pth")
    print(f"Training finished. Final model saved as othello_brain_final_{episodes}.pth")

if __name__ == "__main__":
    train(episodes=50000)