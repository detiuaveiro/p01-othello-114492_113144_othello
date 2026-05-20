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
    classical_agent.set_difficulty("n") if last_winrate < 0.6 else classical_agent.set_difficulty("h")

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
    tau = 0.005

    print(f"Treino iniciado no {device}...")
    start_time = time.time()
    start_time_program = time.time()
    print(time.strftime("%H:%M:%S"))

    for ep in range(episodes):
        # ALTERNÂNCIA: IA é Player 1 nos pares, Player 2 nos ímpares
        ai_player = 1 if ep % 2 == 0 else 2
        opp_player = 3 - ai_player
        
        env.reset()
        done = False
        current_player = 1 # Othello começa sempre pelo Player 1
        is_hard_mode = ep % 5 == 0

        # Variáveis para segurar a transição da IA até que o oponente jogue
        pending_transition = None

        while not done:
            # 1. Obter jogadas válidas para quem tem o turno
            valid_mask = env.get_valid_mask(current_player)
            
            # Se o jogador atual não tem jogadas, passa a vez
            if not any(valid_mask):
                current_player = 3 - current_player
                # Se o próximo também não tiver, o jogo acaba
                if not any(env.get_valid_mask(current_player)):
                    break
                continue

            # --- TURNO DA IA ---
            if current_player == ai_player:
                state = env.get_state(ai_player).to(device)
                
                # Escolha da ação (Epsilon-Greedy)
                if random.random() < epsilon:
                    action = random.choice([i for i, m in enumerate(valid_mask) if m == 1])
                else:
                    with torch.no_grad():
                        q_values = policy_net(state)
                        # MÁSCARA SEGURA: -1e7 para inválidas
                        mask_tensor = torch.FloatTensor(valid_mask).to(device)
                        q_values = torch.where(mask_tensor.bool(), q_values, torch.tensor(-1e7, device=device))
                        action = q_values.argmax().item()

                _, reward, done = env.step(action, ai_player)
                
                # Reward Shaping (Baseado na ação da IA)
                if action in [0, 7, 56, 63]:
                    reward += 2.0
                if action in [1, 8, 9, 6, 14, 15, 48, 49, 57, 62, 55, 54]:
                    reward -= 1.0

                # Se o jogo acabou logo após o movimento da IA
                if done:
                    p1_c = sum(row.count(1) for row in env.board)
                    p2_c = sum(row.count(2) for row in env.board)
                    if (ai_player == 1 and p1_c > p2_c) or (ai_player == 2 and p2_c > p1_c):
                        reward += 10.0
                    elif p1_c == p2_c:
                        reward += 2.0
                    else:
                        reward -= 10.0

                    next_mask = env.get_valid_mask(ai_player)
                    memory.push(state, action, reward, state, next_mask, done) # state repetido apenas como placeholder terminal
                else:
                    # Guardamos a transição pendente (esperando o contra-ataque do oponente)
                    pending_transition = {"state": state, "action": action, "reward": reward}

                current_player = opp_player

            # --- TURNO DO ADVERSÁRIO ---
            else:
                if ep < int(episodes/4):
                    opp_idx = random.choice([i for i, m in enumerate(valid_mask) if m == 1])
                elif is_hard_mode:
                    classical_agent.set_difficulty("n")
                    classical_agent.transposition_table = {}
                    _, move = classical_agent.minmax(env.board, 4, float('-inf'), float('inf'), True, opp_player, classical_agent.use_mobility)
                    opp_idx = move[1] * 8 + move[0]
                else:
                    classical_agent.set_difficulty("n")
                    classical_agent.transposition_table = {}
                    _, move = classical_agent.minmax(env.board, 2, float('-inf'), float('inf'), True, opp_player, classical_agent.use_mobility)
                    opp_idx = move[1] * 8 + move[0]
                
                _, _, done = env.step(opp_idx, opp_player)

                # Se a IA tinha uma jogada pendente, agora sabemos qual é o VERDADEIRO estado seguinte para ela
                if pending_transition is not None:
                    next_state = env.get_state(ai_player).to(device)
                    next_mask = env.get_valid_mask(ai_player)
                    
                    # Se o movimento do oponente encerrou o jogo, calcula recompensa final
                    if done:
                        p1_c = sum(row.count(1) for row in env.board)
                        p2_c = sum(row.count(2) for row in env.board)
                        if (ai_player == 1 and p1_c > p2_c) or (ai_player == 2 and p2_c > p1_c):
                            pending_transition["reward"] += 10.0
                        elif p1_c == p2_c:
                            pending_transition["reward"] += 2.0
                        else:
                            pending_transition["reward"] -= 10.0

                    memory.push(
                        pending_transition["state"],
                        pending_transition["action"],
                        pending_transition["reward"],
                        next_state,
                        next_mask,
                        done
                    )
                    pending_transition = None

                current_player = ai_player

            # --- OTIMIZAÇÃO (BATCH) ---
            if len(memory.buffer) > batch_size and current_player == ai_player: # Otimiza apenas no turno da IA para poupar CPU
                transitions = memory.sample(batch_size)
                b_state = torch.cat([t[0] for t in transitions])
                b_action = torch.tensor([t[1] for t in transitions], device=device).unsqueeze(1)
                b_reward = torch.tensor([t[2] for t in transitions], device=device, dtype=torch.float32)
                b_next_state = torch.cat([t[3] for t in transitions])
                b_next_mask = torch.from_numpy(np.array([t[4] for t in transitions])).to(device).float()
                b_done = torch.tensor([t[5] for t in transitions], device=device, dtype=torch.float32)

                current_q = policy_net(b_state).gather(1, b_action)
                with torch.no_grad():
                    next_q_values = target_net(b_next_state)
                    fill_value = torch.full_like(next_q_values, -1e7)
                    masked_next_q = torch.where(b_next_mask.bool(), next_q_values, fill_value)
                    
                    max_next_q, _ = masked_next_q.max(1)
                    max_next_q = torch.where(b_next_mask.sum(1) > 0, max_next_q, torch.zeros_like(max_next_q))
                    target_q = b_reward + (gamma * max_next_q * (1 - b_done))

                loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                # Soft Update (Polyak Averaging) executado a cada iteração de treino
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

        # --- FIM DO EPISÓDIO ---
        if (ep + 1) % 1000 == 0:
            test_winrate = evaluate_vs_minimax(policy_net, env, classical_agent, device, last_test_winrate)
            last_test_winrate = test_winrate
            if test_winrate >= best_test_winrate:
                best_test_winrate = test_winrate
                torch.save(policy_net.state_dict(), "models/othello_best_strategic.pth")
                print(f"\n[EXAME] Ep {ep+1}: {test_winrate*100:.2f}% vs Minimax (NOVO RECORDE)")

        p1_c = sum(row.count(1) for row in env.board)
        p2_c = sum(row.count(2) for row in env.board)
        if ai_player == 1:
            win_history.append(1 if p1_c > p2_c else 0)
        else:
            win_history.append(1 if p2_c > p1_c else 0)

        epsilon = max(eps_min, epsilon - eps_decay)

        if (ep + 1) % 500 == 0:
            win_rate = sum(win_history) / 100
            if win_rate < 0.35:  
                epsilon = min(0.5, epsilon + 0.15)
                print(time.strftime("%H:%M:%S"))
                print(f"-- RECOVERY: WinRate {win_rate:.2f} baixa, Eps subiu para {epsilon:.2f} --")
        
        if (ep + 1) % 100 == 0:
            curr_time = time.time()
            win_rate = sum(win_history) / len(win_history)
            duration_time=curr_time - start_time_program
            left_time=(duration_time/(ep+1)*(episodes-(ep+1)))*1.2
            hours = int(left_time // 3600)
            minutes = int((left_time % 3600) // 60)
            seconds = int(left_time % 60)
            duration_hours = int(duration_time // 3600)
            duration_minutes = int((duration_time % 3600) // 60)
            duration_seconds = int(duration_time % 60)

            print(
                f"Ep {ep + 1}/{episodes} | Loss: {loss.item():.4f} | WinRate: {win_rate:.2f} | Eps: {epsilon:.2f} | Time: {curr_time - start_time:.1f}s | Est: {hours:02d}h {minutes:02d}m {seconds:02d}s | Dur: {duration_hours:02d}h {duration_minutes:02d}m {duration_seconds:02d}s"
            )
            start_time = curr_time
            
    torch.save(policy_net.state_dict(), "models/othello_brain_final.pth")

if __name__ == "__main__":
    train(episodes=10000)